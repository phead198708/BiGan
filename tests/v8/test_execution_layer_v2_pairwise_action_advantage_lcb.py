from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb import (
    CANDIDATE_NAME,
    ROLE_MARKET_COUNTS,
    PairwiseActionAdvantageLCBPrecollectionFreezeConfig,
    _role_for_rank,
    freeze_pairwise_action_advantage_lcb_precollection,
    validate_pairwise_action_advantage_lcb_feature_contract,
    validate_pairwise_action_advantage_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_pairwise_action_advantage_lcb_fit import (
    _action_advantage_lcb_artifact,
    _apply_action_advantage_lcb_scores,
    _cross_fit_training_predictions,
    _decision_group_ranking_metrics,
    _development_freeze_gate,
    _pairwise_relevance_labels,
    _run_policy_replay,
    _validate_complete_decision_groups,
)
from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from examples.v8.build_execution_layer_v2_pairwise_action_advantage_quarantine_registry import (
    build_quarantine_registry,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)


def test_issue175_protocol_freezes_roles_pairwise_objective_and_quarantine() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    validate_pairwise_action_advantage_lcb_protocol(protocol)

    assert protocol["candidate_name"] == CANDIDATE_NAME
    assert protocol["quarantine_all_issues_through"] == 174
    assert protocol["uses_issue174_confirmatory_labels_for_tuning"] is False
    assert ROLE_MARKET_COUNTS == {
        "development_train": 90,
        "development_calibration": 45,
        "confirmatory_validation": 60,
    }
    assert protocol["cross_fit_protocol"]["objective"] == "rank:pairwise"
    assert protocol["cross_fit_protocol"]["fold_assignment"] == (
        "chronological_expanding_window_prior_markets_only"
    )
    assert protocol["cross_fit_protocol"]["future_market_labels_excluded_from_each_fold"] is True
    assert (
        protocol["action_advantage_lcb_protocol"]["forced_action_side_or_family_quota_enabled"]
        is False
    )
    assert protocol["safety"]["v8_execution_handoff_allowed"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["uses_issue174_confirmatory_labels_for_tuning"] = True
    with pytest.raises(ValueError, match="issue174_quarantined"):
        validate_pairwise_action_advantage_lcb_protocol(drifted)


def test_issue175_feature_contract_is_decision_time_only_and_complete_grid() -> None:
    contract = _load_json(FEATURE_CONTRACT_PATH)
    validate_pairwise_action_advantage_lcb_feature_contract(
        contract,
        expected_parent_protocol_sha256=_sha256(PROTOCOL_PATH),
    )

    assert contract["complete_five_action_decision_grid_required"] is True
    assert contract["decision_group_key_fields"] == ["market_id", "decision_ts"]
    assert contract["action_advantage_against_no_trade_required"] is True
    assert contract["selected_vs_runner_up_advantage_required"] is True
    assert contract["uses_issue174_confirmatory_labels_for_tuning"] is False
    assert contract["settlement_or_outcome_fields_allowed_as_decision_inputs"] is False
    assert "action_no_trade" in contract["feature_columns"]


def test_issue175_role_assignment_is_exact_90_45_60() -> None:
    assert [_role_for_rank(value) for value in (1, 90, 91, 135, 136, 195)] == [
        "development_train",
        "development_train",
        "development_calibration",
        "development_calibration",
        "confirmatory_validation",
        "confirmatory_validation",
    ]
    with pytest.raises(ValueError, match="195-market"):
        _role_for_rank(196)


def test_issue175_precollection_freeze_pins_quarantine_and_role_plan(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        "\n".join(
            [
                json.dumps({"market_id": "prior-a", "decision_ts": 1_000}),
                json.dumps({"market_id": "prior-b", "decision_ts": 2_000}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "issue174-confirmatory.json"
    evidence.write_text('{"confirmatory_gate_passed":false}\n', encoding="utf-8")
    result = freeze_pairwise_action_advantage_lcb_precollection(
        PairwiseActionAdvantageLCBPrecollectionFreezeConfig(
            run_id="issue175-freeze",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            git_commit="a" * 40,
            prior_market_registry_pins=((prior, _sha256(prior)),),
            prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
            expected_prior_unique_market_count=2,
        )
    )
    manifest = result["manifest"]
    assert manifest["target_valid_market_count"] == 195
    assert manifest["role_plan"] == [
        {
            "role": "development_train",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": 90,
        },
        {
            "role": "development_calibration",
            "valid_market_rank_start": 91,
            "valid_market_rank_end": 135,
        },
        {
            "role": "confirmatory_validation",
            "valid_market_rank_start": 136,
            "valid_market_rank_end": 195,
        },
    ]
    assert manifest["collection_started"] is False
    assert manifest["source_model_candidate_eligible"] is False


def test_issue175_quarantine_registry_includes_raw_capture_identity_without_outcomes(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "prior_market_ids": ["prior-a"],
                "maximum_prior_decision_ts": 1_000,
            }
        ),
        encoding="utf-8",
    )
    assignment = tmp_path / "assignment.jsonl"
    assignment.write_text(
        json.dumps(
            {
                "market_id": "assigned-b",
                "maximum_decision_ts": 2_000,
                "labels_or_outcomes_opened_for_assignment": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    capture_run = tmp_path / "capture"
    raw_dir = capture_run / "raw"
    raw_dir.mkdir(parents=True)
    raw_market = raw_dir / "raw_polymarket_markets.jsonl"
    raw_market.write_text(
        json.dumps(
            {
                "market_id": "overflow-c",
                "market_start_ts": 3_000,
                "market_end_ts": 4_000,
                "paper_only": True,
                "capital_at_risk": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch = tmp_path / "batch.json"
    batch.write_text(
        json.dumps(
            {
                "batch_id": "batch",
                "capture_count": 1,
                "captures": [
                    {
                        "run_id": "capture",
                        "run_dir": str(capture_run),
                        "scheduled_round_start_ts": 3_000,
                    }
                ],
                "paper_only": True,
                "capital_at_risk": False,
            }
        ),
        encoding="utf-8",
    )
    result = build_quarantine_registry(
        run_id="issue175-quarantine",
        output_dir=tmp_path / "runs",
        created_at_ts=5_000,
        source_registry_pins=((prior, _sha256(prior)),),
        assignment_rows_pins=((assignment, _sha256(assignment)),),
        batch_progress_pins=((batch, _sha256(batch)),),
    )
    registry = result["registry"]

    assert registry["prior_market_ids"] == [
        "assigned-b",
        "overflow-c",
        "prior-a",
    ]
    assert registry["maximum_prior_decision_ts"] == 4_000
    assert registry["outcome_label_or_pnl_artifacts_opened"] is False
    assert registry["resolution_artifacts_opened"] is False
    assert registry["missing_capture_market_identity_count"] == 0
    assert registry["source_model_candidate_eligible"] is False


def test_pairwise_decision_groups_require_all_five_actions() -> None:
    rows = _decision_rows(market_id="market-a", decision_ts=1_000)
    _validate_complete_decision_groups(rows)
    labels = _pairwise_relevance_labels(sorted(rows, key=lambda row: row["action"]))
    assert sorted(labels) == [0.0, 1.0, 2.0, 3.0, 4.0]

    with pytest.raises(ValueError, match="complete five-action grid"):
        _validate_complete_decision_groups(rows[:-1])


def test_pairwise_ranking_metrics_are_true_decision_point_scoped() -> None:
    first = _decision_rows(market_id="same-market", decision_ts=1_000)
    second = _decision_rows(market_id="same-market", decision_ts=2_000)
    rows = first + second
    scores = [float(row["target_net_pnl_per_contract"]) for row in rows]
    metrics = _decision_group_ranking_metrics(rows, scores)

    assert metrics["decision_group_count"] == 2
    assert metrics["top1_realized_best_action_hit_rate"] == 1.0
    assert metrics["mean_regret"] == 0.0


def test_pairwise_cross_fit_uses_strictly_prior_markets_only() -> None:
    rows = []
    for market_index in range(90):
        market_rows = _decision_rows(
            market_id=f"train-{market_index:03d}",
            decision_ts=1_000 + market_index,
        )
        for action_index, row in enumerate(market_rows):
            row["decision_time_features"] = {"execution_price": float(action_index) / 10.0}
        rows.extend(market_rows)
    model_protocol = dict(_load_json(PROTOCOL_PATH)["cross_fit_protocol"])
    model_protocol["num_boost_round"] = 3
    report = _cross_fit_training_predictions(
        rows,
        feature_columns=("execution_price",),
        model_protocol=model_protocol,
    )

    assert report["objective"] == "rank:pairwise"
    assert report["market_count"] == 90
    assert report["decision_group_count"] == 90
    assert report["initial_training_only_market_count"] == 15
    assert report["oof_market_count"] == 75
    assert report["oof_decision_group_count"] == 75
    assert report["oof_prediction_count"] == 375
    assert report["oof_prediction_coverage_complete"] is True
    assert report["all_development_train_markets_have_oof_predictions"] is False
    assert report["initial_training_markets_excluded_from_oof"] is True
    assert report["future_market_label_access_violation_count"] == 0
    assert [fold["training_market_count"] for fold in report["fold_reports"]] == [
        15,
        30,
        45,
        60,
        75,
    ]
    assert all(
        fold["training_strictly_precedes_validation"] is True
        and fold["training_max_decision_ts"] < fold["validation_min_decision_ts"]
        and fold["future_market_label_access_count"] == 0
        for fold in report["fold_reports"]
    )
    oof_market_ids = {row["market_id"] for row in report["oof_predictions"]}
    assert not ({f"train-{index:03d}" for index in range(15)} & oof_market_ids)
    assert len(oof_market_ids) == 75
    assert report["uses_confirmatory_validation_labels"] is False
    assert report["uses_issue174_confirmatory_labels"] is False


def test_action_advantage_calibration_is_deterministic_and_no_trade_anchored() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    train_oof = []
    calibration = []
    for market_index in range(45):
        rows = _decision_rows(
            market_id=f"cal-{market_index:03d}",
            decision_ts=2_000 + market_index,
        )
        for action_index, row in enumerate(rows):
            row["raw_pairwise_rank_score"] = -0.2 + action_index * 0.1
            calibration.append(row)
    for market_index in range(90):
        rows = _decision_rows(
            market_id=f"train-{market_index:03d}",
            decision_ts=1_000 + market_index,
        )
        for action_index, row in enumerate(rows):
            train_oof.append(
                {
                    "market_id": row["market_id"],
                    "decision_ts": row["decision_ts"],
                    "action": row["action"],
                    "action_family": row["action_family"],
                    "side": row["side"],
                    "action_row_sha256": row["action_row_sha256"],
                    "oof_raw_prediction": -0.2 + action_index * 0.1,
                    "target_net_pnl_per_contract": row["target_net_pnl_per_contract"],
                }
            )

    first = _action_advantage_lcb_artifact(
        calibration,
        train_oof_predictions=train_oof,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    second = _action_advantage_lcb_artifact(
        calibration,
        train_oof_predictions=train_oof,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert set(first["actions"]) == set(REQUIRED_ACTIONS)
    assert first["method"] == ("market_grouped_bootstrap_conditional_action_return_lcb")
    assert first["decision_score_formula"] == ("action_x_oof_score_bucket_target_mean_lcb")
    assert all(
        "calibrated_action_expected_net_return" in group
        and "action_return_lower_confidence_bound" in group
        for group in first["calibration_groups"].values()
    )
    assert first["uses_issue174_confirmatory_labels_for_tuning"] is False

    scored = _apply_action_advantage_lcb_scores(calibration, lcb_artifact=first)
    no_trade = next(row for row in scored if row["action"] == "NO_TRADE")
    assert no_trade["calibrated_action_expected_net_return"] == 0.0
    assert no_trade["action_advantage_lcb_net_return"] == 0.0


def test_development_gate_blocks_before_confirmatory_when_support_is_missing() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    action_rows = [
        row
        for market_index in range(45)
        for row in _decision_rows(
            market_id=f"cal-{market_index:03d}",
            decision_ts=2_000 + market_index,
        )
    ]
    candidate_replay = [
        {
            "execution_guard_order_allowed": True,
            "required_runtime_fields_present": True,
        }
        for _ in range(30)
    ]
    gate = _development_freeze_gate(
        protocol=protocol,
        action_rows=action_rows,
        candidate_replay=candidate_replay,
        candidate_metrics={
            "accepted_bet_count": 30,
            "accepted_unique_market_count": 30,
            "accepted_bet_count_by_side": {"UP": 30, "DOWN": 0},
            "accepted_bet_count_by_family": {
                "HOLD_TO_SETTLEMENT": 30,
                "SELL_BEFORE_CLOSE": 0,
            },
            "net_pnl_sum": 1.0,
            "roi": 0.2,
        },
        baseline_metrics={"net_pnl_sum": 0.5},
        robustness={
            "market_bootstrap_interval_95": {
                "reported": True,
                "lower": 0.1,
            },
            "leave_one_market_out": {
                "reported": True,
                "all_scenarios_positive": True,
            },
            "largest_winner_removal": {
                "reported": True,
                "candidate_net_pnl_after_removal": 0.5,
            },
        },
    )
    assert gate["passed"] is False
    assert "development_side_support_failed" in gate["reason_codes"]
    assert "development_family_support_failed" in gate["reason_codes"]


def test_runner_up_advantage_gate_fails_closed_without_mutating_scores() -> None:
    rows = _decision_rows(market_id="market-a", decision_ts=1_000)
    scores = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.030,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.029,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.010,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.005,
        "NO_TRADE": 0.0,
    }
    predictions = []
    for row in rows:
        score = scores[row["action"]]
        predictions.append(
            {
                **row,
                "raw_pairwise_rank_score": score,
                "calibrated_action_expected_net_return": score,
                "action_advantage_lcb_net_return": score,
            }
        )
    before = canonical_json_sha256(predictions)
    replay = _run_policy_replay(
        predictions,
        score_field="action_advantage_lcb_net_return",
        policy_name=CANDIDATE_NAME,
        entry_threshold=0.02,
        runner_up_advantage_threshold=0.005,
    )

    assert replay[0]["execution_guard_order_allowed"] is False
    assert (
        "selected_vs_runner_up_advantage_not_positive"
        in replay[0]["execution_blocking_reason_codes"]
    )
    assert canonical_json_sha256(predictions) == before
    assert replay[0]["paper_only"] is True
    assert replay[0]["capital_at_risk"] is False


def _decision_rows(*, market_id: str, decision_ts: int) -> list[dict]:
    rows = []
    targets = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.05,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": -0.02,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.03,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.01,
        "NO_TRADE": 0.0,
    }
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if "HOLD_TO_SETTLEMENT" in action
            else "SELL_BEFORE_CLOSE"
            if "SELL_BEFORE_CLOSE" in action
            else "NO_TRADE"
        )
        row = {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": decision_ts + 300_000,
            "max_input_ts": decision_ts,
            "action": action,
            "side": side,
            "action_family": family,
            "decision_time_features": {"execution_price": 0.5 if action != "NO_TRADE" else 0.0},
            "p_up": 0.6,
            "p_down": 0.4,
            "p_up_action_disagreement": bool(side == "DOWN"),
            "microstructure_snapshot": {
                "entry_bid": 0.49,
                "entry_ask": 0.50,
                "spread_bps": 200.0,
                "book_staleness_ms": 100.0,
                "queue_fill_proxy": 0.9,
                "time_to_close_seconds": 180.0,
            },
            "reference_price_feature_provenance": {
                "provenance_valid": True,
                "max_input_ts": decision_ts,
            },
            "target_net_pnl_per_contract": targets[action],
            "target_cost_components": {
                "fees": 0.001,
                "slippage": 0.001,
                "liquidity_impact": 0.001,
            },
            "target_resolved_outcome": "UP",
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "paper_only": True,
            "capital_at_risk": False,
        }
        row["action_row_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
