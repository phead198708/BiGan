from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_future_post_freeze import (
    ChallengeFuturePostFreezeError,
    _frozen_features_by_market,
    build_parallel_settled_targets,
    validate_challenge_future_post_freeze_protocol,
)
from bigan.v8.polymarket.parallel_future_gate import (
    build_parallel_target_free_freeze,
    evaluate_parallel_future_gate,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "v8" / "polymarket_configs"


def _load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text())


def _contracts() -> dict[str, dict]:
    return {
        "v8_1_primary_no_fallback": _load(
            "parallel_candidate_v8_1_primary_no_fallback_contract.json"
        ),
        "v8_3_primary_with_fallback": _load(
            "parallel_candidate_v8_3_primary_with_fallback_contract.json"
        ),
        "matched_frozen_v6_7": _load(
            "parallel_candidate_matched_frozen_v6_7_contract.json"
        ),
    }


def _decision(
    market_id: str,
    decision_ts: int,
    action: str,
    *,
    candidate_id: str,
) -> dict:
    side = "UP" if action.startswith("BUY_UP_") else "DOWN"
    row = {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "policy_decision_ts": decision_ts,
        "executed_action": action,
        "selected_side": side,
        "decision_origin": (
            "matched_v6_7_primary"
            if candidate_id == "matched_frozen_v6_7"
            else "v8_1_primary"
        ),
        "fallback_used": False,
        "primary_abstained": False,
        "execution_guard_order_allowed": True,
        "proposed_order_size": 0.2,
        "target_used_as_decision_input": False,
    }
    if candidate_id == "v8_3_primary_with_fallback":
        row["v8_3_frozen_contract_reproduced"] = True
    if candidate_id == "matched_frozen_v6_7":
        row["matched_baseline_frozen_contract_reproduced"] = True
    return row


def _parallel_freeze(count: int = 40) -> dict:
    source_rows = [
        {
            "schema_version": "bigan-v8-parallel-shared-source-row-v1",
            "market_id": f"market-{index:03d}",
            "decision_ts": 10_000 + index,
            "policy_grid_decision_ts": 10_000 + index,
            "target_used_as_decision_input": False,
        }
        for index in range(count)
    ]
    decisions = {
        "v8_1_primary_no_fallback": [
            _decision(
                row["market_id"],
                row["decision_ts"],
                "BUY_UP_SELL_BEFORE_CLOSE",
                candidate_id="v8_1_primary_no_fallback",
            )
            for row in source_rows
        ],
        "v8_3_primary_with_fallback": [
            _decision(
                row["market_id"],
                row["decision_ts"],
                "BUY_UP_SELL_BEFORE_CLOSE",
                candidate_id="v8_3_primary_with_fallback",
            )
            for row in source_rows
        ],
        "matched_frozen_v6_7": [
            _decision(
                row["market_id"],
                row["decision_ts"],
                "BUY_DOWN_SELL_BEFORE_CLOSE",
                candidate_id="matched_frozen_v6_7",
            )
            for row in source_rows
        ],
    }
    return build_parallel_target_free_freeze(
        protocol=_load("parallel_candidate_protocol.json"),
        candidate_contracts=_contracts(),
        source_rows=source_rows,
        decisions_by_candidate=decisions,
        decision_freeze_created_ts=20_000,
        target_access_started=False,
    )


def _action_targets(freeze: dict) -> list[dict]:
    rows = []
    for source in freeze["shared_source_rows"]:
        for action, pnl, outcome in (
            ("BUY_UP_SELL_BEFORE_CLOSE", 1.0, "UP"),
            ("BUY_DOWN_SELL_BEFORE_CLOSE", -1.0, "UP"),
        ):
            rows.append(
                {
                    "market_id": source["market_id"],
                    "decision_ts": source["policy_grid_decision_ts"],
                    "action": action,
                    "resolved_outcome": outcome,
                    "runtime_policy_after_cost_net_pnl_per_contract": pnl,
                    "cost_fields_subtracted_exactly_once": True,
                    "target_used_as_decision_time_input": False,
                }
            )
    return rows


def test_post_freeze_protocol_is_exactly_pinned() -> None:
    protocol = _load("challenge_future_post_freeze_protocol.json")
    lineage = protocol["lineage"]

    validate_challenge_future_post_freeze_protocol(
        protocol,
        parallel_protocol_sha256=lineage[
            "parallel_candidate_protocol_sha256"
        ],
        collection_plan_sha256=lineage[
            "parallel_future_collection_plan_sha256"
        ],
        frozen_model_binding_sha256=lineage[
            "frozen_model_binding_sha256"
        ],
        runtime_policy_profile_sha256=lineage[
            "runtime_policy_profile_sha256"
        ],
    )

    drifted = json.loads(json.dumps(protocol))
    drifted["target_mapping"]["paper_position_size"] = 0.3
    with pytest.raises(
        ChallengeFuturePostFreezeError,
        match="target_mapping",
    ):
        validate_challenge_future_post_freeze_protocol(
            drifted,
            parallel_protocol_sha256=lineage[
                "parallel_candidate_protocol_sha256"
            ],
            collection_plan_sha256=lineage[
                "parallel_future_collection_plan_sha256"
            ],
            frozen_model_binding_sha256=lineage[
                "frozen_model_binding_sha256"
            ],
            runtime_policy_profile_sha256=lineage[
                "runtime_policy_profile_sha256"
            ],
        )


def test_settled_target_mapping_drives_parallel_hard_gate() -> None:
    freeze = _parallel_freeze()
    targets = build_parallel_settled_targets(
        parallel_freeze=freeze,
        action_runtime_targets=_action_targets(freeze),
    )

    result = evaluate_parallel_future_gate(
        protocol=_load("parallel_candidate_protocol.json"),
        freeze=freeze,
        settled_targets=targets,
        evaluation_started_ts=30_000,
        consumed_freeze_sha256s=set(),
    )

    report = result["report"]
    assert len(targets) == 40
    assert all(
        row["after_cost_pnl_per_notional_by_action"]["NO_TRADE"] == 0.0
        for row in targets
    )
    assert report["candidate_gates"]["v8_1_primary_no_fallback"][
        "all_hard_gates_passed"
    ] is True
    assert report["candidate_gates"]["v8_3_primary_with_fallback"][
        "all_hard_gates_passed"
    ] is True
    assert (
        report["multiplicity_aware_selected_candidate"]
        == "v8_1_primary_no_fallback"
    )


def test_settled_target_mapping_fails_closed_on_missing_frozen_action() -> None:
    freeze = _parallel_freeze(count=1)
    only_up = [
        row
        for row in _action_targets(freeze)
        if row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
    ]

    with pytest.raises(
        ChallengeFuturePostFreezeError,
        match="missing frozen action",
    ):
        build_parallel_settled_targets(
            parallel_freeze=freeze,
            action_runtime_targets=only_up,
        )


def test_frozen_feature_map_rejects_target_or_missing_market() -> None:
    rows = [
        {
            "market_id": "market-001",
            "decision_ts": 10_000,
            "max_input_ts": 9_999,
        },
        {
            "market_id": "market-002",
            "decision_ts": 10_001,
            "max_input_ts": 10_000,
        },
    ]
    grouped = _frozen_features_by_market(
        rows,
        selected_market_ids=["market-001", "market-002"],
    )
    assert set(grouped) == {"market-001", "market-002"}

    rows[0]["resolved_outcome"] = "UP"
    with pytest.raises(
        ChallengeFuturePostFreezeError,
        match="target fields",
    ):
        _frozen_features_by_market(
            rows,
            selected_market_ids=["market-001", "market-002"],
        )
