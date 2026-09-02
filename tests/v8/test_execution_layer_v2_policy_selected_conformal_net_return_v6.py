from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    INDEX_ENTRY_SCHEMA_VERSION,
    ZERO_SHA256,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    PolicySelectedConformalV6DevelopmentSettlementConfig,
    PolicySelectedConformalV6DevelopmentWindowConfig,
    PolicySelectedConformalV6PreRegistrationConfig,
    build_policy_selected_conformal_v6_development_settled_corpus_index,
    build_target_free_v5_no_trade_attrition_report,
    freeze_policy_selected_conformal_v6_development_window,
    pre_register_policy_selected_conformal_v6,
    validate_policy_selected_conformal_v6_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    _target_free_check_support,
    apply_policy_selected_conformal_scores,
    build_policy_selected_conformal_artifact,
    select_sequential_policy_rows,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_policy_selected_conformal_net_return_v6_preregistration_v1.json"
)
COLLECTOR_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
FEATURE_CONTRACT_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
BOUNDARY_TS = 1_784_445_600_000


def test_v6_profile_freezes_policy_selected_calibration_and_future_support() -> None:
    profile = _load_json(PROFILE_PATH)
    validate_policy_selected_conformal_v6_profile(profile)
    assert profile["development_window"]["target_quality_valid_market_count"] == 260
    assert profile["chronological_roles"]["point_model_fit_market_count"] == 150
    assert (
        profile["policy_selected_conformal_calibration"][
            "maximum_selected_trade_rows_per_market"
        ]
        == 1
    )
    assert (
        profile["policy_selected_conformal_calibration"][
            "later_decision_rows_visible_to_earlier_decision"
        ]
        is False
    )
    assert profile["future_evaluation"]["minimum_guard_accepted_unique_market_count"] == 120
    assert profile["future_evaluation"]["minimum_supported_side_market_count"] == 17
    assert profile["safety"] == _blocked_safety_fields()


def test_v6_profile_rejects_calibration_or_safety_relaxation() -> None:
    profile = _load_json(PROFILE_PATH)
    profile["policy_selected_conformal_calibration"][
        "later_decision_rows_visible_to_earlier_decision"
    ] = True
    with pytest.raises(ValueError, match="causal_selection"):
        validate_policy_selected_conformal_v6_profile(profile)


def test_v6_sequential_selection_cannot_use_later_decision_and_selects_once() -> None:
    rows = []
    for decision_ts, up_score in ((100, 0.1), (200, 0.9)):
        for action in (
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
            "NO_TRADE",
        ):
            rows.append(
                {
                    "market_id": "market-1",
                    "decision_ts": decision_ts,
                    "action": action,
                    "side": "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE",
                    "guard_compatible_before_ranking": True,
                    "raw_direct_predicted_net_return": (
                        up_score
                        if action == "BUY_UP_HOLD_TO_SETTLEMENT"
                        else 0.0
                    ),
                }
            )
    selected = select_sequential_policy_rows(
        rows,
        score_field="raw_direct_predicted_net_return",
        require_positive=True,
    )
    assert len(selected) == 1
    assert selected[0]["decision_ts"] == 100
    assert selected[0]["raw_direct_predicted_net_return"] == pytest.approx(0.1)
    assert selected[0]["later_decision_rows_visible_to_selection"] is False


def test_v6_conformal_penalty_uses_only_policy_selected_rows() -> None:
    predictions = []
    targets = []
    actions = (
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    )
    for index in range(50):
        market_id = f"market-{index:03d}"
        selected_action = (
            "BUY_UP_HOLD_TO_SETTLEMENT"
            if index % 2 == 0
            else "BUY_DOWN_HOLD_TO_SETTLEMENT"
        )
        for decision_ts in (100 + index * 10, 101 + index * 10):
            for action in actions:
                score = 0.0
                if action == selected_action:
                    score = 0.2 if decision_ts % 10 == 0 else 0.9
                side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
                row = {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "action": action,
                    "side": side,
                    "guard_compatible_before_ranking": True,
                    "raw_direct_predicted_net_return": score,
                }
                predictions.append(row)
                targets.append(
                    {
                        **row,
                        "target_net_pnl_per_contract": (
                            0.1 if decision_ts % 10 == 0 else -1.0
                        ),
                    }
                )
    artifact = build_policy_selected_conformal_artifact(
        predictions,
        target_rows=targets,
        profile=_load_json(PROFILE_PATH),
        feature_contract_sha256="a" * 64,
    )
    assert artifact["selected_calibration_market_count"] == 50
    assert artifact["selected_side_distribution"] == {"DOWN": 25, "UP": 25}
    assert artifact["sides"]["UP"]["calibration_source"] == "selected_side"
    assert artifact["sides"]["DOWN"]["calibration_source"] == "selected_side"
    assert artifact["sides"]["UP"]["calibration_penalty"] == pytest.approx(0.1)
    assert artifact["sides"]["DOWN"]["calibration_penalty"] == pytest.approx(0.1)
    assert artifact["later_decision_rows_visible_to_selection"] is False


def test_v6_target_free_support_and_score_application_fail_closed() -> None:
    profile = _load_json(PROFILE_PATH)
    selected = [
        {"market_id": f"up-{index}", "side": "UP"} for index in range(5)
    ] + [{"market_id": "down-0", "side": "DOWN"}]
    support = _target_free_check_support(selected, profile=profile)
    assert support["passed"] is False
    assert support["selected_side_market_counts"] == {"UP": 5, "DOWN": 1}

    artifact = {
        "sides": {
            side: {
                "calibration_source": "selected_side",
                "calibration_penalty": 0.1,
            }
            for side in ("UP", "DOWN")
        }
    }
    row = {
        "market_id": "market-1",
        "decision_ts": 100,
        "action": "BUY_UP_HOLD_TO_SETTLEMENT",
        "side": "UP",
        "guard_compatible_before_ranking": True,
        "raw_direct_predicted_net_return": 0.2,
    }
    scored = apply_policy_selected_conformal_scores(
        [row],
        calibration_artifact=artifact,
        profile=profile,
    )
    assert scored[0]["conformal_net_return_lower_bound"] == pytest.approx(0.1)
    assert scored[0]["target_used_as_decision_input"] is False
    assert scored[0]["paper_candidate_allowed"] is False
    with pytest.raises(ValueError, match="target fields"):
        apply_policy_selected_conformal_scores(
            [{**row, "settlement_pnl": 1.0}],
            calibration_artifact=artifact,
            profile=profile,
        )
    profile = _load_json(PROFILE_PATH)
    profile["safety"]["paper_candidate_allowed"] = True
    with pytest.raises(ValueError, match="safety"):
        validate_policy_selected_conformal_v6_profile(profile)


def test_target_free_attrition_explains_positive_raw_scores_becoming_no_trade(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    rows = _prediction_rows()
    _write_jsonl(predictions_path, rows)
    decision_freeze = _decision_freeze(predictions_path)
    report = build_target_free_v5_no_trade_attrition_report(
        decision_freeze,
        prediction_report=_prediction_report(),
        expected_decision_freeze_sha256="a" * 64,
    )
    assert report["decision_group_count"] == 2
    assert report["selected_action_distribution"] == {"NO_TRADE": 2}
    assert report["decision_groups_with_guard_compatible_raw_positive_trade"] == 2
    assert report["decision_groups_with_positive_conformal_trade_lcb"] == 0
    assert report["raw_positive_trade_rows_blocked_by_conformal_penalty"] > 0
    assert report["outcomes_labels_settlement_or_pnl_opened"] is False
    assert report["uses_204_outcomes_for_fitting"] is False
    assert report["paper_candidate_allowed"] is False


def test_target_free_attrition_rejects_outcome_fields(tmp_path: Path) -> None:
    rows = _prediction_rows()
    rows[0]["resolved_outcome"] = "UP"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, rows)
    with pytest.raises(ValueError, match="forbidden_fields"):
        build_target_free_v5_no_trade_attrition_report(
            _decision_freeze(predictions_path),
            prediction_report=_prediction_report(),
            expected_decision_freeze_sha256="a" * 64,
        )


def test_preregistration_freezes_post_issue204_prefix_without_target_access(
    tmp_path: Path,
) -> None:
    fixture = _prereg_fixture(tmp_path)
    result = pre_register_policy_selected_conformal_v6(
        PolicySelectedConformalV6PreRegistrationConfig(
            run_id="issue207-prereg-test",
            output_dir=tmp_path / "runs",
            profile_path=fixture["profile_path"],
            expected_profile_sha256=_sha256(fixture["profile_path"]),
            issue204_window_manifest_path=fixture["window_manifest_path"],
            issue204_decision_freeze_path=fixture["decision_freeze_path"],
            issue204_prediction_report_path=fixture["prediction_report_path"],
            collector_index_path=fixture["index_path"],
            expected_collector_index_prefix_sha256=_sha256(fixture["index_path"]),
            collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
            power_report_path=fixture["power_report_path"],
            power_manifest_path=fixture["power_manifest_path"],
            builder_git_commit="b" * 40,
            preregistration_created_ts=1_784_450_000_000,
        )
    )
    prefix = result["report"]["collector_index_prefix_summary"]
    assert result["report"]["preregistration_passed"] is True
    assert prefix["index_entry_count"] == 240
    assert prefix["quality_valid_index_entry_count"] == 224
    assert prefix["eligible_quality_valid_row_count"] == 4
    assert prefix["eligible_sequence_start"] == 237
    assert prefix["development_markets_remaining"] == 256
    assert result["source_boundary"]["issue204_max_selected_index_sequence"] == 236
    assert result["source_boundary"]["issue204_max_market_end_ts"] == BOUNDARY_TS
    assert result["manifest"]["new_development_target_accessed"] is False
    assert result["manifest"]["future_evaluation_attempted"] is False
    assert result["manifest"]["paper_candidate_allowed"] is False
    assert result["manifest"]["v8_execution_handoff_allowed"] is False


def test_preregistration_rejects_changed_collector_prefix(tmp_path: Path) -> None:
    fixture = _prereg_fixture(tmp_path)
    expected = _sha256(fixture["index_path"])
    fixture["index_path"].write_text(
        fixture["index_path"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collector index prefix SHA-256 mismatch"):
        pre_register_policy_selected_conformal_v6(
            PolicySelectedConformalV6PreRegistrationConfig(
                run_id="issue207-prefix-drift",
                output_dir=tmp_path / "runs",
                profile_path=fixture["profile_path"],
                expected_profile_sha256=_sha256(fixture["profile_path"]),
                issue204_window_manifest_path=fixture["window_manifest_path"],
                issue204_decision_freeze_path=fixture["decision_freeze_path"],
                issue204_prediction_report_path=fixture["prediction_report_path"],
                collector_index_path=fixture["index_path"],
                expected_collector_index_prefix_sha256=expected,
                collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
                power_report_path=fixture["power_report_path"],
                power_manifest_path=fixture["power_manifest_path"],
                builder_git_commit="b" * 40,
                preregistration_created_ts=1_784_450_000_000,
            )
        )


def test_development_window_freezes_earliest_260_markets_and_roles(tmp_path: Path) -> None:
    fixture = _prereg_fixture(tmp_path)
    prereg = pre_register_policy_selected_conformal_v6(
        PolicySelectedConformalV6PreRegistrationConfig(
            run_id="issue207-prereg-for-freeze",
            output_dir=tmp_path / "prereg-runs",
            profile_path=fixture["profile_path"],
            expected_profile_sha256=_sha256(fixture["profile_path"]),
            issue204_window_manifest_path=fixture["window_manifest_path"],
            issue204_decision_freeze_path=fixture["decision_freeze_path"],
            issue204_prediction_report_path=fixture["prediction_report_path"],
            collector_index_path=fixture["index_path"],
            expected_collector_index_prefix_sha256=_sha256(fixture["index_path"]),
            collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
            power_report_path=fixture["power_report_path"],
            power_manifest_path=fixture["power_manifest_path"],
            builder_git_commit="b" * 40,
            preregistration_created_ts=1_784_450_000_000,
        )
    )
    extended_index_path = tmp_path / "extended_index.jsonl"
    _write_jsonl(extended_index_path, _index_rows(total=496))
    result = freeze_policy_selected_conformal_v6_development_window(
        PolicySelectedConformalV6DevelopmentWindowConfig(
            run_id="issue207-development-freeze",
            output_dir=tmp_path / "freeze-runs",
            preregistration_manifest_path=prereg["manifest_path"],
            expected_preregistration_manifest_sha256=prereg["manifest_sha256"],
            collector_index_path=extended_index_path,
            expected_collector_index_sha256=_sha256(extended_index_path),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            builder_git_commit="c" * 40,
            freeze_created_ts=1_784_600_000_000,
        ),
        feature_materializer=_fake_feature_materializer,
        action_materializer=_fake_action_materializer,
    )
    assert result["report"]["development_window_freeze_ready"] is True
    assert result["report"]["selected_market_count"] == 260
    assert result["report"]["scanned_post_boundary_row_count"] == 260
    assert result["report"]["target_free_feature_row_count"] == 1040
    assert result["report"]["target_free_five_action_row_count"] == 5200
    assert result["report"]["selected_sequence_start"] == 237
    assert result["report"]["selected_sequence_end"] == 496
    assert result["report"]["role_market_counts"] == {
        "calibration_check": 50,
        "conformal_calibration": 60,
        "point_model_fit": 150,
    }
    assert result["manifest"]["development_target_accessed"] is False
    assert result["manifest"]["paper_candidate_allowed"] is False
    assert [row["development_role"] for row in result["role_rows"][:150]] == [
        "point_model_fit"
    ] * 150
    assert [row["development_role"] for row in result["role_rows"][-50:]] == [
        "calibration_check"
    ] * 50


def test_development_window_fails_closed_when_support_is_incomplete(tmp_path: Path) -> None:
    fixture = _prereg_fixture(tmp_path)
    prereg = pre_register_policy_selected_conformal_v6(
        PolicySelectedConformalV6PreRegistrationConfig(
            run_id="issue207-prereg-incomplete",
            output_dir=tmp_path / "prereg-runs",
            profile_path=fixture["profile_path"],
            expected_profile_sha256=_sha256(fixture["profile_path"]),
            issue204_window_manifest_path=fixture["window_manifest_path"],
            issue204_decision_freeze_path=fixture["decision_freeze_path"],
            issue204_prediction_report_path=fixture["prediction_report_path"],
            collector_index_path=fixture["index_path"],
            expected_collector_index_prefix_sha256=_sha256(fixture["index_path"]),
            collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
            power_report_path=fixture["power_report_path"],
            power_manifest_path=fixture["power_manifest_path"],
            builder_git_commit="b" * 40,
            preregistration_created_ts=1_784_450_000_000,
        )
    )
    result = freeze_policy_selected_conformal_v6_development_window(
        PolicySelectedConformalV6DevelopmentWindowConfig(
            run_id="issue207-development-incomplete",
            output_dir=tmp_path / "freeze-runs",
            preregistration_manifest_path=prereg["manifest_path"],
            expected_preregistration_manifest_sha256=prereg["manifest_sha256"],
            collector_index_path=fixture["index_path"],
            expected_collector_index_sha256=_sha256(fixture["index_path"]),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            builder_git_commit="c" * 40,
            freeze_created_ts=1_784_600_000_000,
        )
    )
    assert result["report"]["development_window_freeze_ready"] is False
    assert result["report"]["selected_market_count"] == 4
    assert result["report"]["blocking_reason_codes"] == [
        "development_target_quality_valid_market_count_not_met"
    ]
    assert result["manifest"]["selected_rows"] is None
    assert result["manifest"]["development_target_accessed"] is False
    assert result["manifest"]["paper_candidate_allowed"] is False


def test_development_settlement_builds_role_bound_quarantine_index(tmp_path: Path) -> None:
    freeze = _ready_development_freeze(tmp_path)
    result = build_policy_selected_conformal_v6_development_settled_corpus_index(
        PolicySelectedConformalV6DevelopmentSettlementConfig(
            run_id="issue207-development-settlement",
            output_dir=tmp_path / "settlement-runs",
            development_window_manifest_path=freeze["manifest_path"],
            expected_development_window_manifest_sha256=freeze["manifest_sha256"],
            builder_git_commit="d" * 40,
            target_access_started_ts=1_784_700_000_000,
            settlement_max_wait_seconds=0.0,
        ),
        round_finalizer=_fake_round_finalizer,
        monotonic_fn=lambda: 0.0,
    )
    assert result["report"]["development_settled_corpus_ready"] is True
    assert result["report"]["settled_market_count"] == 260
    assert result["report"]["unresolved_market_count"] == 0
    assert result["report"]["policy_pnl_computed"] is False
    assert result["settled_index"]["role_market_counts"] == {
        "calibration_check": 50,
        "conformal_calibration": 60,
        "point_model_fit": 150,
    }
    assert result["manifest"]["source_outcome_blind_rounds_mutated"] is False
    assert result["manifest"]["paper_candidate_allowed"] is False


def test_development_settlement_unresolved_market_blocks_training(tmp_path: Path) -> None:
    freeze = _ready_development_freeze(tmp_path)

    def finalizer_with_one_unresolved(*args: object, **kwargs: object) -> list[dict[str, object]]:
        rows = _fake_round_finalizer(*args, **kwargs)
        market_id = str(rows[-1]["market_id"])
        rows[-1] = {
            "market_id": market_id,
            "settled_corpus_ready": False,
            "failure": {
                "market_id": market_id,
                "pending_resolution": True,
                "reason_codes": ["official_resolution_still_pending"],
            },
        }
        return rows

    result = build_policy_selected_conformal_v6_development_settled_corpus_index(
        PolicySelectedConformalV6DevelopmentSettlementConfig(
            run_id="issue207-development-unresolved",
            output_dir=tmp_path / "settlement-runs",
            development_window_manifest_path=freeze["manifest_path"],
            expected_development_window_manifest_sha256=freeze["manifest_sha256"],
            builder_git_commit="d" * 40,
            target_access_started_ts=1_784_700_000_000,
            settlement_max_wait_seconds=0.0,
        ),
        round_finalizer=finalizer_with_one_unresolved,
        monotonic_fn=lambda: 0.0,
    )
    assert result["report"]["development_settled_corpus_ready"] is False
    assert result["report"]["settled_market_count"] == 259
    assert result["report"]["unresolved_market_count"] == 1
    assert result["report"]["blocking_reason_codes"] == [
        "development_settled_corpus_incomplete"
    ]
    assert result["manifest"]["paper_candidate_allowed"] is False


def _ready_development_freeze(tmp_path: Path) -> dict[str, object]:
    fixture = _prereg_fixture(tmp_path)
    prereg = pre_register_policy_selected_conformal_v6(
        PolicySelectedConformalV6PreRegistrationConfig(
            run_id="issue207-prereg-ready-helper",
            output_dir=tmp_path / "prereg-ready-runs",
            profile_path=fixture["profile_path"],
            expected_profile_sha256=_sha256(fixture["profile_path"]),
            issue204_window_manifest_path=fixture["window_manifest_path"],
            issue204_decision_freeze_path=fixture["decision_freeze_path"],
            issue204_prediction_report_path=fixture["prediction_report_path"],
            collector_index_path=fixture["index_path"],
            expected_collector_index_prefix_sha256=_sha256(fixture["index_path"]),
            collector_protocol_path=COLLECTOR_PROTOCOL_PATH,
            power_report_path=fixture["power_report_path"],
            power_manifest_path=fixture["power_manifest_path"],
            builder_git_commit="b" * 40,
            preregistration_created_ts=1_784_450_000_000,
        )
    )
    extended_index_path = tmp_path / "ready_extended_index.jsonl"
    _write_jsonl(extended_index_path, _index_rows(total=496))
    return freeze_policy_selected_conformal_v6_development_window(
        PolicySelectedConformalV6DevelopmentWindowConfig(
            run_id="issue207-development-ready-helper",
            output_dir=tmp_path / "freeze-ready-runs",
            preregistration_manifest_path=prereg["manifest_path"],
            expected_preregistration_manifest_sha256=prereg["manifest_sha256"],
            collector_index_path=extended_index_path,
            expected_collector_index_sha256=_sha256(extended_index_path),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            builder_git_commit="c" * 40,
            freeze_created_ts=1_784_600_000_000,
        ),
        feature_materializer=_fake_feature_materializer,
        action_materializer=_fake_action_materializer,
    )


def _fake_round_finalizer(
    selected_rows: list[dict[str, object]],
    *,
    run_dir: Path,
    provider_factory: object,
    max_workers: int,
    settlement_attempt: int,
) -> list[dict[str, object]]:
    del provider_factory, max_workers
    results = []
    for selected in selected_rows:
        market_id = str(selected["market_id"])
        corpus_dir = run_dir / "settled_corpus_quarantine" / market_id
        corpus_dir.mkdir(parents=True, exist_ok=True)
        corpus_manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        _write_json(
            corpus_manifest_path,
            {
                "market_id": market_id,
                "settlement_attempt": settlement_attempt,
            },
        )
        results.append(
            {
                "market_id": market_id,
                "settled_corpus_ready": True,
                "index_entry": {
                    "market_id": market_id,
                    "corpus_manifest": _descriptor(corpus_manifest_path),
                    "official_read_only_resolution": True,
                    "source_outcome_blind_round_mutated": False,
                },
            }
        )
    return results


def _prereg_fixture(tmp_path: Path) -> dict[str, Path]:
    index_path = tmp_path / "persistent_index.jsonl"
    index_rows = _index_rows()
    _write_jsonl(index_path, index_rows)
    selected_rows_path = tmp_path / "issue204_selected_rows.jsonl"
    _write_jsonl(selected_rows_path, index_rows[16:236])
    window_manifest_path = tmp_path / "issue204_window_manifest.json"
    _write_json(
        window_manifest_path,
        {
            "window_freeze_ready": True,
            "selected_market_count": 220,
            "labels_outcomes_or_pnl_opened_for_selection": False,
            "selected_rows": _descriptor(selected_rows_path),
            **_blocked_safety_fields(),
        },
    )
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(predictions_path, _prediction_rows())
    decision_freeze_path = tmp_path / "decision_freeze.json"
    _write_json(decision_freeze_path, _decision_freeze(predictions_path))
    prediction_report_path = tmp_path / "prediction_report.json"
    _write_json(prediction_report_path, _prediction_report())
    power_report_path = tmp_path / "power_report.json"
    power_manifest_path = tmp_path / "power_manifest.json"
    _write_json(power_report_path, {"recommended_minimum_accepted_unique_markets": 120})
    _write_json(power_manifest_path, {"uses_204_outcomes_for_planning": False})
    profile = _load_json(PROFILE_PATH)
    profile["frozen_upstream"].update(
        {
            "issue204_window_manifest_sha256": _sha256(window_manifest_path),
            "issue204_decision_freeze_sha256": _sha256(decision_freeze_path),
            "issue204_prediction_report_sha256": _sha256(prediction_report_path),
            "issue205_power_report_sha256": _sha256(power_report_path),
            "issue205_power_manifest_sha256": _sha256(power_manifest_path),
            "collector_protocol_sha256": _sha256(COLLECTOR_PROTOCOL_PATH),
        }
    )
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, profile)
    return {
        "profile_path": profile_path,
        "window_manifest_path": window_manifest_path,
        "decision_freeze_path": decision_freeze_path,
        "prediction_report_path": prediction_report_path,
        "index_path": index_path,
        "power_report_path": power_report_path,
        "power_manifest_path": power_manifest_path,
    }


def _prediction_rows() -> list[dict[str, object]]:
    rows = []
    penalties = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.66,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.71,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.36,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.42,
        "NO_TRADE": 0.0,
    }
    raw = {
        "BUY_UP_HOLD_TO_SETTLEMENT": 0.08,
        "BUY_DOWN_HOLD_TO_SETTLEMENT": 0.04,
        "BUY_UP_SELL_BEFORE_CLOSE": 0.10,
        "BUY_DOWN_SELL_BEFORE_CLOSE": 0.06,
        "NO_TRADE": 0.0,
    }
    for decision_index in range(2):
        for action in penalties:
            compatible = action == "NO_TRADE" or action != "BUY_DOWN_HOLD_TO_SETTLEMENT"
            lower = raw[action] - penalties[action]
            rows.append(
                {
                    "market_id": f"future-market-{decision_index}",
                    "decision_ts": 1_784_440_000_000 + decision_index * 60_000,
                    "action": action,
                    "raw_direct_predicted_net_return": raw[action],
                    "conformal_calibration_penalty": penalties[action],
                    "conformal_net_return_lower_bound": lower,
                    "action_selection_score": lower if compatible else -1_000_000.0,
                    "guard_compatible_before_ranking": compatible,
                    "conformal_calibration_source": (
                        "frozen_no_trade_zero_anchor" if action == "NO_TRADE" else "action"
                    ),
                    "target_used_as_decision_input": False,
                    "target_or_outcome_fields_used": False,
                }
            )
    return rows


def _decision_freeze(predictions_path: Path) -> dict[str, object]:
    return {
        "candidate_target_free_predictions": _descriptor(predictions_path),
        "candidate_guard_accepted_bet_count": 0,
        "future_labels_outcomes_or_pnl_opened": False,
        "target_or_outcome_used_for_decision": False,
        **_blocked_safety_fields(),
    }


def _prediction_report() -> dict[str, object]:
    return {
        "candidate_guard_accepted_bet_count": 0,
        "future_labels_outcomes_or_pnl_opened": False,
        "target_or_outcome_used_for_decision": False,
        **_blocked_safety_fields(),
    }


def _index_rows(*, total: int = 240) -> list[dict[str, object]]:
    rows = []
    previous = ZERO_SHA256
    for sequence in range(1, total + 1):
        market_start = BOUNDARY_TS + (sequence - 237) * 300_000
        row = {
            "schema_version": INDEX_ENTRY_SCHEMA_VERSION,
            "sequence": sequence,
            "previous_entry_sha256": previous,
            "batch_id": f"batch-{(sequence - 1) // 12 + 1:03d}",
            "capture_quality_valid": sequence >= 17,
            "capture_quality_reason_codes": [] if sequence >= 17 else ["synthetic_invalid"],
            "market_id": f"market-{sequence:04d}",
            "slug": f"slug-{sequence:04d}",
            "market_start_ts": market_start,
            "market_end_ts": market_start + 300_000,
            "decision_id": hashlib.sha256(f"decision-{sequence}".encode()).hexdigest(),
            "source_row_hash": hashlib.sha256(f"source-{sequence}".encode()).hexdigest(),
            "labels_outcomes_or_pnl_opened": False,
            **_blocked_safety_fields(),
        }
        row.pop("paper_candidate_allowed")
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = str(row["entry_sha256"])
        rows.append(row)
    return rows


def _fake_feature_materializer(
    selected: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    opened = []
    for selected_row in selected:
        market_id = str(selected_row["market_id"])
        market_start = int(selected_row["market_start_ts"])
        for decision_index in range(4):
            decision_ts = market_start + (decision_index + 1) * 60_000
            rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts - 1,
                }
            )
        opened.append(
            {
                "market_id": market_id,
                "resolution_artifact_opened": False,
            }
        )
    return rows, opened


def _fake_action_materializer(
    feature_rows: list[dict[str, object]],
    *,
    selected_rows: list[dict[str, object]],
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    del selected_rows
    actions = (
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "NO_TRADE",
    )
    rows = []
    for feature in feature_rows:
        for action in actions:
            row = {
                **feature,
                **dict.fromkeys(feature_columns, 0.0),
                "action": action,
                "action_family": (
                    "NO_TRADE"
                    if action == "NO_TRADE"
                    else (
                        "HOLD_TO_SETTLEMENT"
                        if action.endswith("HOLD_TO_SETTLEMENT")
                        else "SELL_BEFORE_CLOSE"
                    )
                ),
                "side": "NONE" if action == "NO_TRADE" else action.split("_")[1],
            }
            row["action_row_sha256"] = canonical_json_sha256(row)
            rows.append(row)
    return rows


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
