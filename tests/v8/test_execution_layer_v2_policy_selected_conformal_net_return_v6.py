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
    PolicySelectedConformalV6PreRegistrationConfig,
    build_target_free_v5_no_trade_attrition_report,
    pre_register_policy_selected_conformal_v6,
    validate_policy_selected_conformal_v6_profile,
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


def _index_rows() -> list[dict[str, object]]:
    rows = []
    previous = ZERO_SHA256
    for sequence in range(1, 241):
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
        row["entry_sha256"] = canonical_json_sha256(row)
        previous = str(row["entry_sha256"])
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
