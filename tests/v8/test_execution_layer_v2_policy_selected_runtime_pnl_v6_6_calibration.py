from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields,
    _descriptor,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_runtime_pnl_v6_6_calibration import (
    SCHEMA_PREFIX,
    _single_use_claim_path,
    _validate_prediction_freeze,
    _validate_target_free_grid,
    _write_single_use_claim,
    build_v6_6_fresh_calibration_artifact,
    build_v6_6_policy_selected_target_free_rows,
    select_exact_v6_6_calibration_index_rows,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_policy_selected_runtime_pnl_v6_6_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _index_row(sequence: int, *, quality: bool = True) -> dict:
    start = 1_784_541_000_000 + (sequence - 528) * 300_000
    return {
        "sequence": sequence,
        "entry_sha256": f"{sequence:064x}"[-64:],
        "previous_entry_sha256": f"{sequence - 1:064x}"[-64:],
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_end_ts": start + 300_000,
        "market_id": f"market-{sequence}",
        "capture_quality_valid": quality,
        "capture_quality_reason_codes": [] if quality else ["synthetic_invalid"],
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
    }


def test_exact_60_selection_is_chronological_outcome_blind_and_disjoint() -> None:
    rows = [_index_row(sequence) for sequence in range(1, 529)]
    rows.extend(_index_row(sequence, quality=sequence != 533) for sequence in range(529, 590))
    selected, attempted = select_exact_v6_6_calibration_index_rows(
        rows,
        profile=_profile(),
        prior_market_ids={"historical-market"},
    )
    assert len(selected) == 60
    assert len(attempted) == 61
    assert selected[0]["sequence"] == 529
    assert selected[-1]["sequence"] == 589
    assert all(row["sequence"] != 533 for row in selected)
    assert all(row["labels_outcomes_or_pnl_opened"] is False for row in selected)


def test_exact_60_selection_fails_closed_on_target_access_or_overlap() -> None:
    rows = [_index_row(sequence) for sequence in range(1, 589)]
    rows[528]["labels_outcomes_or_pnl_opened"] = True
    with pytest.raises(ValueError, match="opened targets"):
        select_exact_v6_6_calibration_index_rows(
            rows, profile=_profile(), prior_market_ids=set()
        )
    rows[528]["labels_outcomes_or_pnl_opened"] = False
    with pytest.raises(ValueError, match="overlaps train"):
        select_exact_v6_6_calibration_index_rows(
            rows,
            profile=_profile(),
            prior_market_ids={"market-529"},
        )


def test_raw_rebuilt_decision_must_match_frozen_market_window() -> None:
    selected = [
        {
            "market_id": "market-a",
            "market_start_ts": 2_000,
            "market_end_ts": 3_000,
        }
    ]
    features = [
        {"market_id": "market-a", "decision_ts": 2_500, "max_input_ts": 2_499}
    ]
    actions = [
        {
            "market_id": "market-a",
            "decision_ts": 2_500,
            "max_input_ts": 2_499,
            "market_close_ts": 3_000,
            "action": action,
        }
        for action in (
            "BUY_UP_HOLD_TO_SETTLEMENT",
            "BUY_DOWN_HOLD_TO_SETTLEMENT",
            "BUY_UP_SELL_BEFORE_CLOSE",
            "BUY_DOWN_SELL_BEFORE_CLOSE",
            "NO_TRADE",
        )
    ]
    _validate_target_free_grid(
        features,
        actions,
        selected_rows=selected,
        minimum_market_start_ts_exclusive=1_500,
    )
    features[0]["decision_ts"] = 1_999
    with pytest.raises(ValueError, match="outside its frozen market window"):
        _validate_target_free_grid(
            features,
            actions,
            selected_rows=selected,
            minimum_market_start_ts_exclusive=1_500,
        )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _prediction_freeze_fixture(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
    profile_path = tmp_path / "profile.json"
    point_path = tmp_path / "point.json"
    selected_path = tmp_path / "selected.jsonl"
    point_rows_path = tmp_path / "point_rows.jsonl"
    decision_path = tmp_path / "decision.json"
    report_path = tmp_path / "report.json"
    _write_json(profile_path, _profile())
    _write_json(point_path, {})
    selected_rows = [
        {"market_id": f"market-{index}"} for index in range(60)
    ]
    point_rows = [
        {
            "market_id": f"market-{index}",
            "side": "UP" if index < 20 else "DOWN",
            "action": (
                "BUY_UP_SELL_BEFORE_CLOSE"
                if index < 20
                else "BUY_DOWN_SELL_BEFORE_CLOSE"
            ),
        }
        for index in range(40)
    ]
    _write_jsonl(selected_path, selected_rows)
    _write_jsonl(point_rows_path, point_rows)
    support = {
        "selected_guard_accepted_sbc_count": 40,
        "count_by_side": {"UP": 20, "DOWN": 20},
        "minimum_required_per_side": 20,
        "target_free_support_gate_passed": True,
        "blocking_reason_codes": [],
    }
    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-freeze-v1",
        "future_target_access_allowed": True,
        "target_free_support": support,
        "selected_market_count": 60,
        "selected_market_ids": [row["market_id"] for row in selected_rows],
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        **_blocked_safety_fields(),
    }
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-report-v1",
        "target_free_support_gate_passed": True,
        "future_target_access_allowed": True,
        "selected_market_count": 60,
        "policy_selected_guard_accepted_sbc_count": 40,
        "policy_selected_guard_accepted_sbc_count_by_side": {"UP": 20, "DOWN": 20},
        "target_free_support_blocking_reason_codes": [],
        "labels_outcomes_resolution_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    _write_json(decision_path, decision)
    _write_json(report_path, report)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-prediction-freeze-manifest-v1",
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "resolution_artifact_opened": False,
        "settlement_provider_called": False,
        "profile": _descriptor(profile_path),
        "point_freeze_manifest": _descriptor(point_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "selected_window_rows": _descriptor(selected_path),
        "point_predictions": _descriptor(point_rows_path),
        **_blocked_safety_fields(),
    }
    return manifest, profile_path, point_path, decision_path, report_path


def test_prediction_freeze_target_access_requires_cross_artifact_agreement(
    tmp_path: Path,
) -> None:
    manifest, profile_path, point_path, _, _ = _prediction_freeze_fixture(tmp_path)
    _validate_prediction_freeze(
        manifest,
        profile_path=profile_path,
        point_path=point_path,
    )


@pytest.mark.parametrize("artifact_name", ["decision", "report"])
def test_prediction_freeze_blocks_semantically_inconsistent_referenced_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    manifest, profile_path, point_path, decision_path, report_path = (
        _prediction_freeze_fixture(tmp_path)
    )
    artifact_path = decision_path if artifact_name == "decision" else report_path
    payload = json.loads(artifact_path.read_text())
    payload["future_target_access_allowed"] = False
    _write_json(artifact_path, payload)
    descriptor_field = (
        "accepted_bet_decision_freeze" if artifact_name == "decision" else "report"
    )
    manifest[descriptor_field] = _descriptor(artifact_path)
    with pytest.raises(ValueError, match="target-access mismatch"):
        _validate_prediction_freeze(
            manifest,
            profile_path=profile_path,
            point_path=point_path,
        )


def test_prediction_freeze_recomputes_support_from_frozen_point_rows(
    tmp_path: Path,
) -> None:
    manifest, profile_path, point_path, _, _ = _prediction_freeze_fixture(tmp_path)
    point_rows_path = Path(manifest["point_predictions"]["path"])
    point_rows = [json.loads(line) for line in point_rows_path.read_text().splitlines()]
    point_rows[0]["side"] = "DOWN"
    point_rows[0]["action"] = "BUY_DOWN_SELL_BEFORE_CLOSE"
    _write_jsonl(point_rows_path, point_rows)
    manifest["point_predictions"] = _descriptor(point_rows_path)
    with pytest.raises(ValueError, match="target-access mismatch"):
        _validate_prediction_freeze(
            manifest,
            profile_path=profile_path,
            point_path=point_path,
        )


def test_policy_selected_mapper_uses_only_guard_accepted_sbc() -> None:
    market_id = "market-a"
    decision_ts = 1_900_000_000_000
    action = "BUY_UP_SELL_BEFORE_CLOSE"
    predictions = [
        {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "market_close_ts": decision_ts + 240_000,
            "max_input_ts": decision_ts - 1,
            "action": action,
            "decision_time_features": {
                "execution_price": 0.61,
                "selected_side_probability": 0.64,
            },
            "microstructure_snapshot": {
                "entry_bid": 0.60,
                "spread_bps": 165.0,
                "queue_fill_proxy": 0.91,
                "time_to_close_seconds": 240.0,
            },
        }
    ]
    replay = [
        {
            "market_id": market_id,
            "decision_ts": decision_ts,
            "executed_action": action,
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "selected_side": "UP",
            "decision_score": 0.07,
            "execution_guard_order_allowed": True,
            "viability_row_sha256": "a" * 64,
        },
        {
            "market_id": "blocked",
            "decision_ts": decision_ts + 1,
            "executed_action": "BUY_DOWN_SELL_BEFORE_CLOSE",
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "selected_side": "DOWN",
            "decision_score": 0.08,
            "execution_guard_order_allowed": False,
            "viability_row_sha256": "b" * 64,
        },
    ]
    rows = build_v6_6_policy_selected_target_free_rows(replay, predictions=predictions)
    assert len(rows) == 1
    assert rows[0]["features"] == {
        "side_is_up": 1.0,
        "execution_price": 0.61,
        "current_bid": 0.6,
        "spread_bps": 165.0,
        "queue_fill_probability_proxy": 0.91,
        "time_to_close_seconds": 240.0,
        "selected_side_probability": 0.64,
        "canonical_v6_2_score": 0.07,
    }
    assert rows[0]["target_fields_used_for_selection"] is False
    assert rows[0]["target_fields_used_as_model_inputs"] is False


def _joined_rows(*, up_count: int = 20, down_count: int = 20) -> list[dict]:
    rows = []
    for side, count in (("UP", up_count), ("DOWN", down_count)):
        for index in range(count):
            target = 0.10 + index * 0.002
            point = target + (0.001 if index % 2 else -0.001)
            rows.append(
                {
                    "market_id": f"{side}-{index}",
                    "decision_ts": 2_000_000_000_000 + len(rows),
                    "max_input_ts": 1_999_999_999_999 + len(rows),
                    "side": side,
                    "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                    "features": {"canonical_v6_2_score": -0.25},
                    "runtime_expected_net_pnl_point": point,
                    "runtime_policy_after_cost_net_pnl_per_contract": target,
                    "point_residual": point - target,
                }
            )
    return rows


def _train_rows() -> list[dict]:
    return [
        {"runtime_policy_after_cost_net_pnl_per_contract": -0.2},
        {"runtime_policy_after_cost_net_pnl_per_contract": 0.2},
    ]


def test_fresh_calibration_passes_only_with_both_side_support_and_better_errors() -> None:
    artifact = build_v6_6_fresh_calibration_artifact(
        _joined_rows(),
        train_rows=_train_rows(),
        profile=_profile(),
        point_model_descriptor={"path": "/tmp/model", "sha256": "1" * 64},
        decision_freeze_descriptor={"path": "/tmp/freeze", "sha256": "2" * 64},
        settled_index_descriptor={"path": "/tmp/index", "sha256": "3" * 64},
        runtime_policy_profile_descriptor={"path": "/tmp/runtime", "sha256": "4" * 64},
    )
    assert artifact["fresh_calibration_gate_passed"] is True
    assert artifact["positive_lcb_selected_market_count_by_side"] == {
        "UP": 20,
        "DOWN": 20,
    }
    assert artifact["fresh_calibration_outcomes_used_as_model_inputs"] is False
    assert artifact["future_unseen_side_only_pnl_gate_required"] is True
    assert artifact["source_model_candidate_eligible"] is False
    assert artifact["promotion_evidence_eligible"] is False


def test_fresh_calibration_fails_closed_when_up_support_is_insufficient() -> None:
    artifact = build_v6_6_fresh_calibration_artifact(
        _joined_rows(up_count=19),
        train_rows=_train_rows(),
        profile=_profile(),
        point_model_descriptor={"path": "/tmp/model", "sha256": "1" * 64},
        decision_freeze_descriptor={"path": "/tmp/freeze", "sha256": "2" * 64},
        settled_index_descriptor={"path": "/tmp/index", "sha256": "3" * 64},
        runtime_policy_profile_descriptor={"path": "/tmp/runtime", "sha256": "4" * 64},
    )
    assert artifact["fresh_calibration_gate_passed"] is False
    assert "side_calibration_support_gate_failed" in artifact[
        "fresh_calibration_gate_blocking_reason_codes"
    ]
    assert artifact["freeze_ready"] is False


def test_fresh_calibration_single_use_claim_is_fail_closed(tmp_path: Path) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}\n")
    claim_path = _single_use_claim_path(freeze_path)
    claim = {"claim_id": "once"}
    _write_single_use_claim(claim_path, claim)
    with pytest.raises(ValueError, match="already been consumed"):
        _write_single_use_claim(claim_path, claim)
