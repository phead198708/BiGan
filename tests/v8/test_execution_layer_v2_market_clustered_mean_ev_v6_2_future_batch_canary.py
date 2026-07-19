from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary import (
    MarketClusteredMeanEVV62FutureBatchCanaryConfig,
    build_v6_2_future_cumulative_canary,
    run_market_clustered_mean_ev_v6_2_future_batch_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
    _blocked_safety_fields,
)


def test_future_batch_scores_both_sides_and_preserves_target_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _input_fixture(tmp_path, decision_ts=2_000)
    module = __import__(
        "bigan.v8.polymarket.training."
        "execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary",
        fromlist=["unused"],
    )
    monkeypatch.setattr(module.xgb.Booster, "load_model", lambda self, path: None)
    monkeypatch.setattr(
        module,
        "_raw_target_stripped_predictions",
        lambda booster, rows, feature_columns: [dict(row) for row in rows],
    )
    monkeypatch.setattr(
        module,
        "attach_frozen_execution_compatibility",
        lambda rows: [
            {**row, "guard_compatible_before_ranking": row["action"] != "NO_TRADE"}
            for row in rows
        ],
    )

    def fake_scores(rows, *, calibration_artifact):
        output = []
        for row in rows:
            selected = (
                row["market_id"] == "future-up"
                and row["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
            ) or (
                row["market_id"] == "future-down"
                and row["action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
            )
            output.append(
                {
                    **row,
                    "mean_ev_lower_confidence_bound": 0.05 if selected else 0.0,
                    "action_advantage_lcb_net_return": 0.05 if selected else 0.0,
                    "raw_direct_predicted_net_return": 0.06 if selected else 0.0,
                }
            )
        return output

    monkeypatch.setattr(module, "apply_market_clustered_mean_ev_scores", fake_scores)
    monkeypatch.setattr(
        module,
        "_outcome_blind_acceptance_replay",
        lambda rows, **kwargs: [
            {
                "market_id": row["market_id"],
                "selected_side": row["side"],
                "executed_action": row["action"],
                "execution_guard_order_allowed": True,
                "execution_blocking_reason_codes": [],
            }
            for row in rows
            if float(row["mean_ev_lower_confidence_bound"]) > 0.0
        ],
    )
    result = run_market_clustered_mean_ev_v6_2_future_batch_canary(
        MarketClusteredMeanEVV62FutureBatchCanaryConfig(
            run_id="future-batch",
            output_dir=tmp_path / "out",
            development_batch_canary_manifest_path=inputs["development_manifest"],
            expected_development_batch_canary_manifest_sha256=_sha256(
                inputs["development_manifest"]
            ),
            candidate_manifest_path=inputs["candidate_manifest"],
            expected_candidate_manifest_sha256=_sha256(inputs["candidate_manifest"]),
        )
    )
    report = result["report"]
    assert report["future_strictly_later_and_disjoint_passed"] is True
    assert report["positive_mean_ev_lcb_side_market_count"] == {"UP": 1, "DOWN": 1}
    assert report["guard_accepted_by_side"] == {"UP": 1, "DOWN": 1}
    assert report["labels_outcomes_or_pnl_opened"] is False
    assert report["promotion_evidence_eligible"] is False


def test_future_batch_fails_closed_before_freeze_boundary(tmp_path: Path) -> None:
    inputs = _input_fixture(tmp_path, decision_ts=1_000)
    with pytest.raises(ValueError, match="strictly_later_or_causal"):
        run_market_clustered_mean_ev_v6_2_future_batch_canary(
            MarketClusteredMeanEVV62FutureBatchCanaryConfig(
                run_id="pre-freeze",
                output_dir=tmp_path / "out",
                development_batch_canary_manifest_path=inputs["development_manifest"],
                expected_development_batch_canary_manifest_sha256=_sha256(
                    inputs["development_manifest"]
                ),
                candidate_manifest_path=inputs["candidate_manifest"],
                expected_candidate_manifest_sha256=_sha256(
                    inputs["candidate_manifest"]
                ),
            )
        )


def test_cumulative_canary_counts_unique_side_support_and_completes_at_200() -> None:
    reports = []
    for index in range(20):
        quality = 10
        up = [f"up-{index:02d}-{item}" for item in range(3)]
        down = [f"down-{index:02d}-{item}" for item in range(3)]
        reports.append(_batch_report(index + 1, quality=quality, up=up, down=down))
    report = build_v6_2_future_cumulative_canary(reports, run_id="complete")
    assert report["quality_valid_market_count"] == 200
    assert report["guard_accepted_unique_market_count"] == 120
    assert report["guard_accepted_unique_market_count_by_side"] == {
        "UP": 60,
        "DOWN": 60,
    }
    assert report["future_holdout_collection_complete"] is True
    assert report["collector_should_stop"] is True
    assert report["target_free_terminal_blocked"] is False
    assert report["future_pnl_evaluation_allowed"] is False


def test_two_complete_zero_action_batches_stop_before_outcome_access() -> None:
    report = build_v6_2_future_cumulative_canary(
        [
            _batch_report(1, quality=12, up=[], down=[]),
            _batch_report(2, quality=12, up=[], down=[]),
        ],
        run_id="zero",
    )
    assert report["target_free_terminal_blocked"] is True
    assert "two_consecutive_complete_batches_zero_v6_2_actions" in report[
        "target_free_terminal_blocking_reason_codes"
    ]
    assert report["labels_outcomes_or_pnl_opened"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False


def _input_fixture(tmp_path: Path, *, decision_ts: int) -> dict[str, Path]:
    feature_contract = tmp_path / "feature_contract.json"
    _write_json(feature_contract, {"feature_columns": ["feature"]})
    model = tmp_path / "model.json"
    model.write_text("{}\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    _write_json(
        calibration,
        {
            "frozen": True,
            "calibration_gate_passed": True,
            "sides": {
                "UP": {"mean_residual": 0.0, "mean_residual_upper_confidence_bound": 0.0},
                "DOWN": {
                    "mean_residual": 0.0,
                    "mean_residual_upper_confidence_bound": 0.0,
                },
            },
        },
    )
    v5_train = tmp_path / "v5_train.jsonl"
    v5_calibration = tmp_path / "v5_calibration.jsonl"
    issue209_rows = tmp_path / "issue209_rows.jsonl"
    _write_jsonl(v5_train, [{"market_id": "old-train"}])
    _write_jsonl(v5_calibration, [{"market_id": "old-calibration"}])
    _write_jsonl(issue209_rows, [{"market_id": "old-target-free"}])
    v5_manifest = tmp_path / "v5_manifest.json"
    _write_json(
        v5_manifest,
        {
            "development_train_action_rows": _descriptor(v5_train),
            "development_calibration_action_rows": _descriptor(v5_calibration),
        },
    )
    issue209_manifest = tmp_path / "issue209_manifest.json"
    _write_json(
        issue209_manifest,
        {"target_free_five_action_rows": _descriptor(issue209_rows)},
    )
    pre_audit = tmp_path / "pre_audit.json"
    _write_json(
        pre_audit,
        {
            "feature_contract": _descriptor(feature_contract),
            "v5_freeze_manifest": _descriptor(v5_manifest),
        },
    )
    candidate_manifest = tmp_path / "candidate_manifest.json"
    _write_json(
        candidate_manifest,
        {
            "candidate_name": "market_clustered_mean_ev_v6_2",
            "target_free_actionability_gate_passed": True,
            "research_actionability_candidate_frozen": True,
            "collector_resume_allowed": True,
            "new_strictly_later_future_holdout_required": True,
            "future_collection_minimum_created_ts_exclusive": 1_000,
            "target_free_labels_outcomes_settlement_targets_or_pnl_opened": False,
            "pre_target_access_audit": _descriptor(pre_audit),
            "source_issue209_manifest": _descriptor(issue209_manifest),
            "source_model": _descriptor(model),
            "market_clustered_mean_risk_calibration": _descriptor(calibration),
            **_blocked_safety_fields(),
        },
    )
    action_rows = []
    for market_id in ("future-up", "future-down"):
        for action in REQUIRED_ACTIONS:
            side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
            family = (
                "SELL_BEFORE_CLOSE"
                if action.endswith("SELL_BEFORE_CLOSE")
                else "HOLD_TO_SETTLEMENT"
                if action.endswith("HOLD_TO_SETTLEMENT")
                else "NO_TRADE"
            )
            action_rows.append(
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts,
                    "action": action,
                    "side": side,
                    "action_family": family,
                    "feature": 1.0,
                }
            )
    action_path = tmp_path / "actions.jsonl"
    _write_jsonl(action_path, action_rows)
    development_report = tmp_path / "development_report.json"
    _write_json(
        development_report,
        {
            "development_data_canary_passed": True,
            "bounded_batch_complete": True,
            "source_sequence_start": 313,
            "source_sequence_end": 314,
            "indexed_market_count": 2,
        },
    )
    development_manifest = tmp_path / "development_manifest.json"
    _write_json(
        development_manifest,
        {
            "batch_id": "batch-27",
            "development_data_canary_passed": True,
            "labels_outcomes_or_pnl_opened": False,
            "report": _descriptor(development_report),
            "five_action_grid": _descriptor(action_path),
        },
    )
    return {
        "candidate_manifest": candidate_manifest,
        "development_manifest": development_manifest,
    }


def _batch_report(
    index: int, *, quality: int, up: list[str], down: list[str]
) -> dict[str, object]:
    accepted = sorted(set(up + down))
    start = (index - 1) * 12 + 313
    return {
        "batch_id": f"batch-{index:03d}",
        "candidate_name": "market_clustered_mean_ev_v6_2",
        "future_strictly_later_and_disjoint_passed": True,
        "bounded_batch_complete": True,
        "source_sequence_start": start,
        "source_sequence_end": start + 11,
        "indexed_market_count": 12,
        "quality_valid_market_count": quality,
        "positive_guard_compatible_trade_lcb_row_count": len(accepted),
        "positive_mean_ev_lcb_unique_market_count": len(accepted),
        "guard_accepted_unique_market_count": len(accepted),
        "guard_accepted_market_ids": accepted,
        "guard_accepted_market_ids_by_side": {"UP": up, "DOWN": down},
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
