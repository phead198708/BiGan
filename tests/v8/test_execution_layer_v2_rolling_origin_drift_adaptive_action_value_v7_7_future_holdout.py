from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout import (
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
    SCAN_CAP,
    STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE,
    _safety_fields,
    build_v7_7_future_pnl_noninferiority_gate,
    build_v7_7_target_free_holdout_freeze_report,
    materialize_guard_accepted_runtime_decisions,
    select_v7_7_future_holdout_window,
    validate_v7_7_future_holdout_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_pipeline import (
    V77FutureTargetFreeFreezeConfig,
    _baseline_guard_window,
    _load_and_validate_excluded_attempts,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_plan.json"
)


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text())


def _target_row(index: int, pnl: float, *, side: str = "UP") -> dict:
    return {
        "market_id": f"market-{index:03d}",
        "decision_ts": 1_000_000 + index,
        "max_input_ts": 999_000 + index,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
        "target_available_only_post_exit_or_official_resolution": True,
        "target_used_as_decision_time_input": False,
    }


def _market_ids() -> list[str]:
    return [f"market-{index:03d}" for index in range(120)]


def _index_row(index: int, *, quality_valid: bool = True) -> dict:
    market_start_ts = STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE + 300_000 * (
        index + 1
    )
    return {
        "sequence": index + 1,
        "market_id": f"market-{index:03d}",
        "slug": f"market-slug-{index:03d}",
        "decision_id": f"decision-{index:03d}",
        "source_row_hash": f"{index + 1:064x}",
        "scheduled_round_start_ts": market_start_ts,
        "market_start_ts": market_start_ts,
        "market_end_ts": market_start_ts + 300_000,
        "capture_quality_valid": quality_valid,
    }


def _action_rows() -> list[dict]:
    actions = (
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    )
    return [
        {
            "market_id": f"market-{index:03d}",
            "decision_ts": 2_000_000 + index,
            "max_input_ts": 1_999_000 + index,
            "action": action,
        }
        for index in range(120)
        for action in actions
    ]


def _guard_rows(*, accepted_count: int, side: str = "DOWN") -> list[dict]:
    return [
        {
            "market_id": f"market-{index:03d}",
            "decision_ts": 2_000_000 + index,
            "selected_action": f"BUY_{side}_SELL_BEFORE_CLOSE"
            if index < accepted_count
            else "NO_TRADE",
            "selected_side": side if index < accepted_count else "NONE",
            "execution_guard_order_allowed": index < accepted_count,
            "source_score_mutated": False,
            "labels_outcomes_or_pnl_opened": False,
        }
        for index in range(120)
    ]


def test_plan_freezes_bounded_strictly_later_outcome_blind_collection() -> None:
    plan = _plan()

    validate_v7_7_future_holdout_plan(plan)
    assert _sha256_file(PLAN_PATH) == FROZEN_PLAN_SHA256

    collection = plan["collection"]
    assert collection["exact_quality_valid_market_count"] == EXACT_MARKET_COUNT == 120
    assert collection["maximum_attempted_market_count"] == SCAN_CAP == 180
    assert (
        collection["strictly_later_minimum_market_start_ts_exclusive"]
        == STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE
        == 1_784_760_900_000
    )
    assert collection["outcomes_resolution_labels_or_pnl_opened"] is False
    assert collection["candidate_model_scoring_during_collection_allowed"] is False


def test_plan_uses_inclusive_noninferiority_without_side_quota() -> None:
    plan = _plan()
    freeze = plan["target_free_decision_freeze"]
    gate = plan["single_use_future_pnl_gate"]

    assert freeze["minimum_v7_7_guard_accepted_unique_market_count"] == (
        MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
    )
    assert freeze["side_quota_enabled"] is False
    assert gate["comparison_operator"] == "greater_than_or_equal"
    assert gate["equality_passes_noninferiority"] is True
    assert gate["candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"] == 0.0
    assert (
        gate[
            "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl_minimum_inclusive"
        ]
        == 0.0
    )
    assert gate["candidate_total_after_cost_pnl_minimum_exclusive"] == 0.0


def test_plan_rejects_outcome_access_or_gate_drift() -> None:
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["collection"]["outcomes_resolution_labels_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="collection"):
        validate_v7_7_future_holdout_plan(changed)

    changed = copy.deepcopy(plan)
    changed["single_use_future_pnl_gate"][
        "candidate_minus_v6_7_total_after_cost_pnl_minimum_inclusive"
    ] = -0.01
    with pytest.raises(ValueError, match="single_use_gate"):
        validate_v7_7_future_holdout_plan(changed)


def test_plan_safety_remains_fail_closed() -> None:
    assert _plan()["safety"] == _safety_fields()
    assert _safety_fields() == {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
    }


def test_equal_positive_candidate_passes_inclusive_noninferiority() -> None:
    rows = [_target_row(index, 0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        rows,
        baseline_rows=[dict(row) for row in rows],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="a" * 64,
    )

    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["future_noninferiority_gate_passed"] is True
    assert report["future_pnl_gate_passed"] is True
    assert report["model_improvement_demonstrated"] is False
    assert report["promotion_discussion_evidence_available"] is True
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_distinct_candidate_actions_can_pass_without_side_quota() -> None:
    candidate = [_target_row(index, 0.02, side="DOWN") for index in range(40)]
    baseline = [_target_row(index, 0.01, side="UP") for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="b" * 64,
    )

    assert report["candidate_side_distribution_diagnostic"] == {"DOWN": 40}
    assert report["v6_7_side_distribution_diagnostic"] == {"UP": 40}
    assert report["side_quota_enabled"] is False
    assert report["candidate_minus_v6_7_after_cost_pnl"] == pytest.approx(0.4)
    assert report["future_pnl_gate_passed"] is True


def test_equal_negative_candidate_fails_only_absolute_pnl_check() -> None:
    rows = [_target_row(index, -0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        rows,
        baseline_rows=[dict(row) for row in rows],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="c" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is True
    assert report["future_pnl_gate_passed"] is False
    assert report["future_pnl_gate_blocking_reason_codes"] == [
        "candidate_total_after_cost_pnl_not_positive"
    ]


def test_inferior_candidate_fails_total_noninferiority() -> None:
    candidate = [_target_row(index, 0.009) for index in range(40)]
    baseline = [_target_row(index, 0.01) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="d" * 64,
    )

    assert report["future_noninferiority_gate_passed"] is False
    assert "candidate_total_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_largest_winner_removed_noninferiority_is_hard_gate() -> None:
    candidate = [_target_row(index, 0.01) for index in range(40)]
    candidate[0] = _target_row(0, 1.0)
    baseline = [_target_row(index, 0.02) for index in range(40)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=baseline,
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="e" * 64,
    )

    assert report["candidate_minus_v6_7_after_cost_pnl"] > 0.0
    assert report[
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl"
    ] < 0.0
    assert report["future_noninferiority_gate_passed"] is False
    assert "candidate_largest_winner_removed_pnl_inferior_to_v6_7" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]


def test_support_and_complete_settlement_remain_fail_closed() -> None:
    candidate = [_target_row(index, 0.01) for index in range(39)]
    report = build_v7_7_future_pnl_noninferiority_gate(
        candidate,
        baseline_rows=[dict(row) for row in candidate],
        evaluation_market_ids=_market_ids(),
        settled_market_ids=_market_ids(),
        plan=_plan(),
        target_free_freeze_sha256="f" * 64,
    )
    assert "insufficient_v7_7_guard_accepted_unique_market_support" in report[
        "future_pnl_gate_blocking_reason_codes"
    ]

    with pytest.raises(ValueError, match="exact settled"):
        build_v7_7_future_pnl_noninferiority_gate(
            [_target_row(index, 0.01) for index in range(40)],
            baseline_rows=[_target_row(index, 0.01) for index in range(40)],
            evaluation_market_ids=_market_ids(),
            settled_market_ids=_market_ids()[:-1],
            plan=_plan(),
            target_free_freeze_sha256="0" * 64,
        )


def test_window_selection_uses_earliest_exact_120_within_scan_cap() -> None:
    rows = [_index_row(index) for index in range(130)]
    selected, attempted, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids=set(),
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(selected) == 120
    assert len(attempted) == 130
    assert [row["sequence"] for row in selected] == list(range(1, 121))
    assert summary["exact_window_ready"] is True


def test_window_selection_excludes_invalid_and_prior_identity_rows() -> None:
    rows = [_index_row(index) for index in range(122)]
    rows[0]["capture_quality_valid"] = False
    selected, _, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids={"market-001"},
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(selected) == 120
    assert selected[0]["market_id"] == "market-002"
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1,
        "prior_market_id_overlap": 1,
    }


def test_scan_cap_counts_failed_attempts_with_missing_market_start() -> None:
    rows = [_index_row(index) for index in range(SCAN_CAP + 1)]
    for index in range(61):
        rows[index]["capture_quality_valid"] = False
        rows[index]["market_start_ts"] = 0
    selected, attempted, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids=set(),
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(attempted) == SCAN_CAP
    assert len(selected) == 119
    assert rows[SCAN_CAP]["capture_quality_valid"] is True
    assert rows[SCAN_CAP] not in attempted
    assert summary["exact_window_ready"] is False
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 61,
        "market_start_not_strictly_later": 61,
    }


def test_scan_cap_allows_exact_120_only_within_first_180_attempts() -> None:
    rows = [_index_row(index) for index in range(SCAN_CAP)]
    for index in range(60):
        rows[index]["capture_quality_valid"] = False
        rows[index]["market_start_ts"] = 0
    selected, attempted, summary = select_v7_7_future_holdout_window(
        rows,
        plan=_plan(),
        prior_market_ids=set(),
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(attempted) == SCAN_CAP
    assert len(selected) == EXACT_MARKET_COUNT
    assert selected[0]["sequence"] == 61
    assert selected[-1]["sequence"] == SCAN_CAP
    assert summary["exact_window_ready"] is True


def test_target_free_freeze_passes_one_sided_support_without_outcomes() -> None:
    selected = [_index_row(index) for index in range(120)]
    report = build_v7_7_target_free_holdout_freeze_report(
        selected,
        attempted_rows=selected,
        action_rows=_action_rows(),
        candidate_guard_rows=_guard_rows(accepted_count=40, side="DOWN"),
        baseline_guard_rows=_guard_rows(accepted_count=50, side="UP"),
        selection_summary={"exact_window_ready": True},
        plan=_plan(),
        stage_started_ts=max(row["market_end_ts"] for row in selected) + 1,
        collector_index_sha256="1" * 64,
    )

    assert report["target_free_freeze_passed"] is True
    assert report["v7_7_guard_accepted_market_count"] == 40
    assert report["v7_7_guard_accepted_side_distribution_diagnostic"] == {"DOWN": 40}
    assert report["side_quota_enabled"] is False
    assert report["future_target_access_allowed"] is True
    assert report["labels_outcomes_resolution_or_pnl_opened"] is False


def test_target_free_freeze_fails_support_causality_and_target_leakage() -> None:
    selected = [_index_row(index) for index in range(120)]
    actions = _action_rows()
    actions[0]["max_input_ts"] = actions[0]["decision_ts"] + 1
    actions[1]["settlement_pnl"] = 1.0
    report = build_v7_7_target_free_holdout_freeze_report(
        selected,
        attempted_rows=selected,
        action_rows=actions,
        candidate_guard_rows=_guard_rows(accepted_count=39),
        baseline_guard_rows=_guard_rows(accepted_count=50),
        selection_summary={"exact_window_ready": True},
        plan=_plan(),
        stage_started_ts=max(row["market_end_ts"] for row in selected) + 1,
        collector_index_sha256="2" * 64,
    )

    assert report["target_free_freeze_passed"] is False
    assert set(report["target_free_blocking_reason_codes"]) >= {
        "target_free_v7_7_guard_accepted_support_insufficient",
        "target_free_five_action_grid_incomplete",
        "target_free_feature_causality_violation",
        "target_free_forbidden_target_field_present",
    }


def test_guard_accepted_runtime_decisions_bind_frozen_source_rows() -> None:
    actions = _action_rows()
    for row in actions:
        row["market_close_ts"] = row["decision_ts"] + 60_000
        row["decision_id"] = f"{row['market_id']}:{row['action']}"
        row["microstructure_snapshot"] = {"time_to_close_seconds": 60.0}
    accepted = materialize_guard_accepted_runtime_decisions(
        _guard_rows(accepted_count=40, side="DOWN"),
        action_rows=actions,
    )

    assert len(accepted) == 40
    assert {row["side"] for row in accepted} == {"DOWN"}
    assert {row["action"] for row in accepted} == {
        "BUY_DOWN_SELL_BEFORE_CLOSE"
    }
    assert all(row["max_input_ts"] <= row["decision_ts"] for row in accepted)
    assert all(row["source_score_mutated"] is False for row in accepted)
    assert all(
        row["labels_outcomes_resolution_or_pnl_opened"] is False
        for row in accepted
    )


def test_guard_accepted_runtime_decisions_bind_exact_decision_timestamp() -> None:
    actions = _action_rows()
    selected = next(
        row
        for row in actions
        if row["market_id"] == "market-000"
        and row["action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    )
    selected["decision_id"] = "selected-decision"
    selected["market_close_ts"] = selected["decision_ts"] + 60_000
    selected["microstructure_snapshot"] = {"time_to_close_seconds": 60.0}
    actions.append(
        {
            **selected,
            "decision_ts": selected["decision_ts"] - 1_000,
            "max_input_ts": selected["max_input_ts"] - 1_000,
            "decision_id": "earlier-decision",
        }
    )

    accepted = materialize_guard_accepted_runtime_decisions(
        _guard_rows(accepted_count=1, side="DOWN"),
        action_rows=actions,
    )

    assert len(accepted) == 1
    assert accepted[0]["decision_ts"] == selected["decision_ts"]
    assert accepted[0]["source_decision_id"] == "selected-decision"


def test_guard_accepted_runtime_decisions_fail_without_exact_decision_timestamp() -> None:
    actions = _action_rows()
    guards = _guard_rows(accepted_count=1, side="DOWN")
    guards[0]["decision_ts"] += 1

    with pytest.raises(ValueError, match="source identity"):
        materialize_guard_accepted_runtime_decisions(
            guards,
            action_rows=actions,
        )


def test_guard_accepted_runtime_decisions_fail_on_missing_source_action() -> None:
    actions = [
        row
        for row in _action_rows()
        if not (
            row["market_id"] == "market-000"
            and row["action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
        )
    ]
    with pytest.raises(ValueError, match="source identity"):
        materialize_guard_accepted_runtime_decisions(
            _guard_rows(accepted_count=40, side="DOWN"),
            action_rows=actions,
        )


def test_target_free_pipeline_config_requires_aligned_pinned_batch_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="nonempty and aligned"):
        V77FutureTargetFreeFreezeConfig(
            run_id="freeze",
            output_dir=tmp_path,
            plan_path=PLAN_PATH,
            expected_plan_sha256=FROZEN_PLAN_SHA256,
            collector_protocol_path=tmp_path / "protocol.json",
            expected_collector_protocol_sha256="1" * 64,
            collector_index_path=tmp_path / "index.jsonl",
            expected_collector_index_sha256="2" * 64,
            excluded_attempt_rows_path=tmp_path / "excluded.jsonl",
            expected_excluded_attempt_rows_sha256="7" * 64,
            historical_manifest_path=tmp_path / "historical.json",
            expected_historical_manifest_sha256="3" * 64,
            prior_lineage_rows_path=tmp_path / "lineage.jsonl",
            expected_prior_lineage_rows_sha256="4" * 64,
            prior_canary_index_path=tmp_path / "canary.jsonl",
            expected_prior_canary_index_sha256="5" * 64,
            development_batch_manifest_paths=(tmp_path / "development.json",),
            expected_development_batch_manifest_sha256s=("6" * 64,),
            v6_2_batch_manifest_paths=(),
            expected_v6_2_batch_manifest_sha256s=(),
            implementation_commit="a" * 40,
            stage_started_ts=1,
        )


def test_v6_7_baseline_guard_replay_is_complete_and_fail_closed() -> None:
    profile = json.loads(
        (
            ROOT
            / "examples/v8/polymarket_configs/"
            "execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json"
        ).read_text()
    )
    source = {
        "market_id": "market-1",
        "decision_ts": 2_000,
        "max_input_ts": 1_999,
        "action": "BUY_DOWN_SELL_BEFORE_CLOSE",
        "decision_time_features": {
            "execution_price": 0.55,
            "selected_side_executable_ask_notional": 1.0,
            "selected_side_executable_bid_notional": 1.0,
            "selected_side_liquidity_depth": 1.0,
        },
        "microstructure_snapshot": {
            "spread_bps": 10.0,
            "book_staleness_ms": 10.0,
            "queue_fill_proxy": 1.0,
            "time_to_close_seconds": 120.0,
        },
        "reference_price_feature_provenance": {"provenance_valid": True},
    }
    rows = _baseline_guard_window(
        ["market-1", "market-2"],
        baseline_rows=[
            {
                "market_id": "market-1",
                "decision_ts": 2_000,
                "action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            }
        ],
        action_rows=[source],
        v6_7_profile=profile,
    )

    assert len(rows) == 2
    assert rows[0]["execution_guard_order_allowed"] is True
    assert rows[0]["selected_side"] == "DOWN"
    assert rows[1]["selected_action"] == "NO_TRADE"
    assert rows[1]["execution_guard_order_allowed"] is False
    assert rows[1]["execution_blocking_reason_codes"] == [
        "v6_7_no_positive_guard_compatible_action"
    ]
    assert all(row["source_score_mutated"] is False for row in rows)
    assert all(row["labels_outcomes_or_pnl_opened"] is False for row in rows)


def test_unindexed_failed_attempt_registry_is_target_free_and_consumes_scan_cap(
    tmp_path: Path,
) -> None:
    capture_report_path = tmp_path / "pending_round_capture_report.json"
    attempt_id = "partial-round-1"
    capture_safety = {
        "run_id": attempt_id,
        "capture_status": "blocked_fail_closed",
        "pending_resolution": False,
        "resolution_provider_called": False,
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "live_exchange_write_enabled": False,
        "broker_exchange_write_enabled": False,
    }
    capture_report_path.write_text(
        json.dumps(
            {
                **capture_safety,
                "training_eligible": False,
                "raw_resolution_count": 0,
                "public_collection_reason_codes": [
                    "read_only_public_http_transport_error"
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    capture_manifest_path = tmp_path / "pending_round_capture_manifest.json"
    capture_manifest_path.write_text(
        json.dumps(capture_safety, sort_keys=True) + "\n"
    )
    excluded_path = tmp_path / "excluded.jsonl"
    excluded = {
        "schema_version": (
            "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-"
            "future-holdout-excluded-collection-attempt-v1"
        ),
        "attempt_id": attempt_id,
        "run_id": attempt_id,
        "scheduled_round_start_ts": (
            STRICTLY_LATER_MINIMUM_MARKET_START_TS_EXCLUSIVE + 1
        ),
        "market_start_ts": 0,
        "capture_quality_valid": False,
        "capture_quality_reason_codes": ["read_only_public_http_transport_error"],
        "excluded_from_selection": True,
        "excluded_from_settlement": True,
        "excluded_from_quality_valid_support": True,
        "counts_against_frozen_scan_cap": True,
        "pending_round_capture_report": _descriptor(capture_report_path),
        "pending_round_capture_manifest": _descriptor(capture_manifest_path),
        "labels_outcomes_or_pnl_opened": False,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        **_safety_fields(),
    }
    excluded["excluded_attempt_row_id"] = canonical_json_sha256(excluded)
    excluded_path.write_text(json.dumps(excluded, sort_keys=True) + "\n")

    rows = _load_and_validate_excluded_attempts(excluded_path, plan=_plan())
    assert rows == [excluded]

    index_rows = [_index_row(index) for index in range(SCAN_CAP)]
    selected, attempted, summary = select_v7_7_future_holdout_window(
        [*index_rows, *rows],
        plan=_plan(),
        prior_market_ids=set(),
        prior_slugs=set(),
        prior_decision_ids=set(),
        prior_source_row_hashes=set(),
    )

    assert len(attempted) == SCAN_CAP
    assert rows[0] in attempted
    assert index_rows[-1] not in attempted
    assert len(selected) == EXACT_MARKET_COUNT
    assert summary["exclusion_reason_distribution"] == {
        "capture_quality_invalid": 1,
        "market_start_not_strictly_later": 1,
    }
