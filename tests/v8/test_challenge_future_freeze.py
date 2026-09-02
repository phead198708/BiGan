from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_future_freeze import (
    ChallengeFutureFreezeError,
    build_challenge_parallel_decisions,
    build_parallel_shared_source_rows,
    resolve_challenge_collection_service_root,
    select_challenge_future_window,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.parallel_future_gate import (
    build_parallel_target_free_freeze,
)
from examples.v8.run_challenge_future_freeze import _status

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "examples" / "v8" / "polymarket_configs"


def test_service_root_is_derived_exactly_from_frozen_plan() -> None:
    plan_path = CONFIG_DIR / "parallel_future_collection_plan.json"
    plan = json.loads(plan_path.read_text())
    expected = ROOT / str(plan["collection"]["service_root"])

    assert resolve_challenge_collection_service_root(
        collection_plan=plan,
        collection_plan_path=plan_path,
    ) == expected.resolve()
    assert resolve_challenge_collection_service_root(
        collection_plan=plan,
        collection_plan_path=plan_path,
        requested_service_root=expected,
    ) == expected.resolve()


def test_service_root_rejects_stale_or_same_suffix_checkout(
    tmp_path: Path,
) -> None:
    plan_path = CONFIG_DIR / "parallel_future_collection_plan.json"
    plan = json.loads(plan_path.read_text())
    same_suffix = tmp_path / str(plan["collection"]["service_root"])

    with pytest.raises(
        ChallengeFutureFreezeError,
        match="does not match",
    ):
        resolve_challenge_collection_service_root(
            collection_plan=plan,
            collection_plan_path=plan_path,
            requested_service_root=same_suffix,
        )

    invalid = copy.deepcopy(plan)
    invalid["collection"]["service_root"] = "../outside"
    with pytest.raises(
        ChallengeFutureFreezeError,
        match="service root is invalid",
    ):
        resolve_challenge_collection_service_root(
            collection_plan=invalid,
            collection_plan_path=plan_path,
        )


def test_status_default_uses_plan_derived_service_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    config_dir = repository / "examples/v8/polymarket_configs"
    config_dir.mkdir(parents=True)
    plan = json.loads(
        (CONFIG_DIR / "parallel_future_collection_plan.json").read_text()
    )
    plan_path = config_dir / "parallel_future_collection_plan.json"
    plan_bytes = (
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    plan_path.write_bytes(plan_bytes)
    plan_path.with_suffix(".sha256").write_text(
        hashlib.sha256(plan_bytes).hexdigest() + "\n",
        encoding="ascii",
    )

    status = _status(
        argparse.Namespace(
            plan=plan_path,
            service_root=None,
        )
    )

    expected = repository / str(plan["collection"]["service_root"])
    assert status["service_root"] == str(expected.resolve())
    assert status["collector_index_exists"] is False
    assert status["selected_market_count"] == 0
    assert status["service_status"] == "not_started"
    assert status["collection_started"] is False
    assert status["operator_collection_authorization_required"] is True
    assert (
        status["operator_collection_authorization_granted_at_refreeze"]
        is False
    )


def _index_row(
    sequence: int,
    *,
    boundary: int = 1_000,
    quality_valid: bool = True,
) -> dict:
    market_id = f"market-{sequence:03d}"
    slug = f"btc-updown-{sequence:03d}"
    decision_id = canonical_json_sha256(
        {"market_id": market_id, "sequence": sequence}
    )
    source_hash = canonical_json_sha256(
        {"market_id": market_id, "source": sequence}
    )
    return {
        "sequence": sequence,
        "run_id": f"run-{sequence:03d}",
        "batch_id": f"batch-{(sequence - 1) // 12:03d}",
        "scheduled_round_start_ts": boundary + sequence,
        "market_start_ts": boundary + sequence,
        "market_end_ts": boundary + sequence + 300_000,
        "market_id": market_id,
        "slug": slug,
        "decision_id": decision_id,
        "source_row_hash": source_hash,
        "entry_sha256": canonical_json_sha256(
            {"entry": sequence}
        ),
        "capture_quality_valid": quality_valid,
        "raw_artifacts": {
            "raw.jsonl": {
                "path": f"/tmp/{market_id}.jsonl",
                "sha256": canonical_json_sha256(
                    {"raw": sequence}
                ),
                "row_count": 1,
            }
        },
    }


def _plan(*, boundary: int = 1_000) -> dict:
    return {
        "collection": {
            "strictly_later_minimum_market_start_ts_exclusive": boundary,
            "quality_valid_market_target": 120,
            "maximum_attempted_market_count": 180,
        }
    }


def _guard(
    market_id: str,
    *,
    decision_ts: int,
    action: str,
    allowed: bool,
) -> dict:
    side = (
        "UP"
        if action.startswith("BUY_UP_")
        else "DOWN"
        if action.startswith("BUY_DOWN_")
        else "NONE"
    )
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "execution_blocking_reason_codes": (
            [] if allowed else ["policy_abstention"]
        ),
        "full_execution_guard_unchanged": True,
    }


def test_select_challenge_future_window_uses_earliest_exact_120() -> None:
    rows = [
        _index_row(1, boundary=0),
        _index_row(2, quality_valid=False),
        *[_index_row(sequence) for sequence in range(3, 125)],
    ]
    rows[0]["scheduled_round_start_ts"] = 1_000
    rows[0]["market_start_ts"] = 1_000

    selected, attempted, summary = select_challenge_future_window(
        list(reversed(rows)),
        collection_plan=_plan(),
    )

    assert len(attempted) == 124
    assert len(selected) == 120
    assert selected[0]["sequence"] == 3
    assert selected[-1]["sequence"] == 122
    assert summary["exact_window_ready"] is True
    assert summary["strictly_later_time_violation_count"] == 0
    assert summary["selected_identity_duplicate_count"] == 0
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1,
        "market_start_not_strictly_later": 1,
        "scheduled_round_not_strictly_later": 1,
    }


def test_select_challenge_future_window_reports_terminal_cap() -> None:
    rows = [
        _index_row(sequence, quality_valid=sequence > 100)
        for sequence in range(1, 181)
    ]

    selected, attempted, summary = select_challenge_future_window(
        rows,
        collection_plan=_plan(),
    )

    assert len(attempted) == 180
    assert len(selected) == 80
    assert summary["exact_window_ready"] is False
    assert summary["attempt_cap_exhausted"] is True
    assert summary["remaining_quality_valid_market_count"] == 40


def test_shared_source_rows_fall_back_to_scheduled_timestamp() -> None:
    selected = [_index_row(1), _index_row(2)]
    baseline = [
        _guard(
            "market-001",
            decision_ts=2_001,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            allowed=True,
        ),
        _guard(
            "market-002",
            decision_ts=0,
            action="NO_TRADE",
            allowed=False,
        ),
    ]

    rows = build_parallel_shared_source_rows(
        selected,
        baseline_guard_rows=baseline,
    )

    assert [row["decision_ts"] for row in rows] == [2_001, 1_002]
    assert rows[0]["policy_grid_decision_ts"] == 2_001
    assert "policy_grid_decision_ts" not in rows[1]
    assert all(row["target_used_as_decision_input"] is False for row in rows)


def test_parallel_decision_projection_passes_gate_contract() -> None:
    source_rows = [
        {
            "schema_version": "bigan-v8-parallel-shared-source-row-v1",
            "market_id": "market-001",
            "decision_ts": 2_001,
            "target_used_as_decision_input": False,
        },
        {
            "schema_version": "bigan-v8-parallel-shared-source-row-v1",
            "market_id": "market-002",
            "decision_ts": 2_002,
            "target_used_as_decision_input": False,
        },
    ]
    v8_1 = [
        _guard(
            "market-001",
            decision_ts=2_001,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            allowed=True,
        ),
        _guard(
            "market-002",
            decision_ts=2_002,
            action="NO_TRADE",
            allowed=False,
        ),
    ]
    v6_7 = [
        _guard(
            "market-001",
            decision_ts=2_001,
            action="BUY_DOWN_SELL_BEFORE_CLOSE",
            allowed=True,
        ),
        _guard(
            "market-002",
            decision_ts=2_002,
            action="BUY_UP_SELL_BEFORE_CLOSE",
            allowed=True,
        ),
    ]
    v8_3 = [
        {
            **v8_1[0],
            "selection_source": "v8_1_primary",
            "selection_reason_codes": ["v8_1_primary_full_guard_passed"],
            "fallback_applied": False,
            "original_v8_1_action": "BUY_UP_SELL_BEFORE_CLOSE",
        },
        {
            **v6_7[1],
            "selection_source": "v6_7_non_risk_abstention_fallback",
            "selection_reason_codes": [
                "v8_1_policy_level_non_risk_abstention",
                "v6_7_independent_full_guard_passed",
            ],
            "fallback_applied": True,
            "original_v8_1_action": "NO_TRADE",
        },
    ]
    projected = build_challenge_parallel_decisions(
        source_rows,
        v8_1_guard_rows=v8_1,
        v8_3_overlay_rows=v8_3,
        v6_7_guard_rows=v6_7,
        position_size=0.2,
    )
    protocol = json.loads(
        (CONFIG_DIR / "parallel_candidate_protocol.json").read_text()
    )
    contracts = {
        "v8_1_primary_no_fallback": json.loads(
            (
                CONFIG_DIR
                / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
            ).read_text()
        ),
        "v8_3_primary_with_fallback": json.loads(
            (
                CONFIG_DIR
                / "parallel_candidate_v8_3_primary_with_fallback_contract.json"
            ).read_text()
        ),
        "matched_frozen_v6_7": json.loads(
            (
                CONFIG_DIR
                / "parallel_candidate_matched_frozen_v6_7_contract.json"
            ).read_text()
        ),
    }

    freeze = build_parallel_target_free_freeze(
        protocol=protocol,
        candidate_contracts=contracts,
        source_rows=source_rows,
        decisions_by_candidate=projected,
        decision_freeze_created_ts=3_000,
        target_access_started=False,
    )

    assert freeze["shared_source_row_count"] == 2
    assert projected["v8_1_primary_no_fallback"][1][
        "executed_action"
    ] == "NO_TRADE"
    assert projected["v8_1_primary_no_fallback"][1][
        "fallback_used"
    ] is False
    assert projected["v8_3_primary_with_fallback"][1][
        "fallback_used"
    ] is True
    assert projected["v8_3_primary_with_fallback"][1][
        "primary_abstained"
    ] is True


def test_parallel_decision_projection_rejects_missing_market() -> None:
    with pytest.raises(
        ChallengeFutureFreezeError,
        match="v8.3 row missing",
    ):
        build_challenge_parallel_decisions(
            [
                {
                    "market_id": "market-001",
                    "decision_ts": 2_001,
                }
            ],
            v8_1_guard_rows=[
                _guard(
                    "market-001",
                    decision_ts=2_001,
                    action="NO_TRADE",
                    allowed=False,
                )
            ],
            v8_3_overlay_rows=[],
            v6_7_guard_rows=[
                _guard(
                    "market-001",
                    decision_ts=2_001,
                    action="NO_TRADE",
                    allowed=False,
                )
            ],
            position_size=0.2,
        )
