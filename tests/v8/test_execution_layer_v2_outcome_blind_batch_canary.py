from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_evaluation import (
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    OutcomeBlindDevelopmentBatchCanaryConfig,
    build_frozen_model_cumulative_canary,
    build_v5_retrospective_no_trade_canary_report,
    run_outcome_blind_development_batch_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    INDEX_ENTRY_SCHEMA_VERSION,
    ZERO_SHA256,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    REQUIRED_ACTIONS,
)


def test_development_batch_canary_materializes_target_free_grid_and_guard_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch_id = "batch-001"
    summary_path = tmp_path / "batch_summary.json"
    _write_json(
        summary_path,
        {
            "batch_id": batch_id,
            "captures": [{"run_id": "round-1"}],
            "errors": [],
            "outcome_blind_collection_only": True,
            "settlement_finalizer_started": False,
            "resolution_provider_called": False,
            "training_corpus_export_attempted": False,
            "labels_or_outcomes_opened_during_collection": False,
            "settlement_pnl_opened_during_collection": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        },
    )
    index_path = tmp_path / "index.jsonl"
    pending_manifest_path = tmp_path / "pending_manifest.json"
    pending_report_path = tmp_path / "pending_report.json"
    resolution_path = tmp_path / "raw_polymarket_resolutions.jsonl"
    _write_json(pending_manifest_path, {"pending_resolution": True})
    _write_json(pending_report_path, {"status": "captured"})
    resolution_path.write_text("", encoding="utf-8")
    entry = {
        "schema_version": INDEX_ENTRY_SCHEMA_VERSION,
        "sequence": 1,
        "previous_entry_sha256": ZERO_SHA256,
        "batch_id": batch_id,
        "scheduled_round_start_ts": 100,
        "market_start_ts": 100,
        "market_end_ts": 400,
        "market_id": "market-1",
        "slug": "slug-1",
        "decision_id": "1" * 64,
        "source_row_hash": "2" * 64,
        "capture_quality_valid": True,
        "capture_quality_reason_codes": [],
        "duplicate_identity_reason_codes": [],
        "batch_summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        "pending_round_capture_manifest": {
            "path": str(pending_manifest_path),
            "sha256": _sha256(pending_manifest_path),
        },
        "pending_round_capture_report": {
            "path": str(pending_report_path),
            "sha256": _sha256(pending_report_path),
        },
        "raw_artifacts": {
            "raw_polymarket_resolutions.jsonl": {
                "path": str(resolution_path),
                "sha256": _sha256(resolution_path),
                "row_count": 0,
            }
        },
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    entry["entry_sha256"] = canonical_json_sha256(entry)
    index_path.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    feature_contract_path = tmp_path / "feature_contract.json"
    _write_json(feature_contract_path, {"feature_columns": ["feature_x"]})

    feature_rows = [
        {"market_id": "market-1", "decision_ts": 200, "max_input_ts": 199}
    ]
    action_rows = []
    for action in REQUIRED_ACTIONS:
        side = "UP" if "BUY_UP" in action else "DOWN" if "BUY_DOWN" in action else "NONE"
        family = (
            "HOLD_TO_SETTLEMENT"
            if "HOLD_TO_SETTLEMENT" in action
            else "SELL_BEFORE_CLOSE"
            if "SELL_BEFORE_CLOSE" in action
            else "NO_TRADE"
        )
        action_rows.append(
            {
                "market_id": "market-1",
                "decision_ts": 200,
                "max_input_ts": 199,
                "action": action,
                "side": side,
                "action_family": family,
                "feature_x": 1.0,
            }
        )
    universe_rows = []
    for row in action_rows:
        allowed = row["action"] == "BUY_UP_HOLD_TO_SETTLEMENT"
        universe_rows.append(
            {
                **row,
                "p_up_alignment_passed": allowed,
                "execution_quality_only_passed": allowed,
                "full_guard_original_action_allowed": allowed,
                "execution_blocking_reason_codes": [] if allowed else ["static_guard_blocked"],
            }
        )

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary."
        "_materialize_selected_window_features",
        lambda rows: (feature_rows, [{"raw_feature_artifacts": {}}]),
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary."
        "_materialize_future_action_rows",
        lambda rows, **kwargs: action_rows,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary."
        "build_execution_compatible_action_universe",
        lambda rows: universe_rows,
    )

    result = run_outcome_blind_development_batch_canary(
        OutcomeBlindDevelopmentBatchCanaryConfig(
            run_id="canary",
            output_dir=tmp_path / "runs",
            collector_index_path=index_path,
            expected_collector_index_sha256=_sha256(index_path),
            batch_id=batch_id,
            feature_contract_path=feature_contract_path,
            expected_feature_contract_sha256=_sha256(feature_contract_path),
        )
    )
    report = result["report"]
    assert report["bounded_batch_complete"] is True
    assert report["complete_five_action_grid_passed"] is True
    assert report["static_full_guard_original_action_allowed_count"] == 1
    assert report["candidate_model_scoring_attempted"] is False
    assert report["labels_outcomes_or_pnl_opened"] is False
    assert report["development_data_canary_passed"] is True
    assert result["manifest"]["paper_candidate_allowed"] is False


def test_frozen_model_cumulative_canary_blocks_three_complete_zero_signal_batches() -> None:
    reports = [_batch_report(index, positive=0, accepted={}) for index in range(1, 4)]
    report = build_frozen_model_cumulative_canary(reports, run_id="zero-three")
    assert report["consecutive_zero_signal_batch_count"] == 3
    assert report["consecutive_zero_signal_quality_market_count"] == 36
    assert report["target_free_terminal_blocked"] is True
    assert (
        "three_consecutive_complete_batches_zero_positive_lcb_and_guard_acceptance"
        in report["target_free_terminal_blocking_reason_codes"]
    )
    assert report["labels_outcomes_or_pnl_opened"] is False


def test_one_weak_batch_remains_diagnostic_and_capacity_early_stop_is_fail_closed() -> None:
    one = build_frozen_model_cumulative_canary(
        [_batch_report(1, positive=0, accepted={})], run_id="one"
    )
    assert one["target_free_terminal_blocked"] is False
    capacity = build_frozen_model_cumulative_canary(
        [
            _batch_report(
                1,
                positive=2,
                accepted={"UP": ["market-up"], "DOWN": ["market-down"]},
            )
        ],
        run_id="capacity",
        minimum_accepted_market_count=5,
        minimum_side_market_count=3,
        maximum_index_scan_count=13,
    )
    assert capacity["remaining_maximum_market_capacity"] == 1
    assert capacity["target_free_terminal_blocked"] is True
    assert "remaining_scan_capacity_cannot_reach_minimum_accepted_support" in capacity[
        "target_free_terminal_blocking_reason_codes"
    ]
    assert "remaining_scan_capacity_cannot_reach_up_side_support" in capacity[
        "target_free_terminal_blocking_reason_codes"
    ]


def test_v5_retrospective_detects_target_free_stop_after_36_markets() -> None:
    rows = []
    for market_index in range(36):
        for action in REQUIRED_ACTIONS:
            rows.append(
                {
                    "future_window_selection_rank": market_index + 1,
                    "market_id": f"market-{market_index:03d}",
                    "decision_ts": 1000 + market_index,
                    "action": action,
                    "guard_compatible_before_ranking": True,
                    "conformal_net_return_lower_bound": 0.0,
                    "execution_guard_order_allowed": False,
                }
            )
    report = build_v5_retrospective_no_trade_canary_report(rows, run_id="v5")
    assert report["first_target_free_terminal_stop_after_batch"] == 3
    assert report["first_target_free_terminal_stop_after_market_count"] == 36
    assert report["v5_would_have_been_blocked_earlier"] is True
    assert report["retrospective_changes_historical_v5_result"] is False


def test_v5_retrospective_rejects_outcome_fields() -> None:
    with pytest.raises(ValueError, match="forbidden targets"):
        build_v5_retrospective_no_trade_canary_report(
            [
                {
                    "future_window_selection_rank": 1,
                    "market_id": "market",
                    "decision_ts": 100,
                    "action": "NO_TRADE",
                    "settlement_pnl": 1.0,
                }
            ],
            run_id="invalid",
        )


def _batch_report(
    index: int, *, positive: int, accepted: dict[str, list[str]]
) -> dict[str, object]:
    accepted_ids = sorted({market_id for values in accepted.values() for market_id in values})
    return {
        "batch_id": f"batch-{index:03d}",
        "bounded_batch_complete": True,
        "source_sequence_start": (index - 1) * 12 + 1,
        "source_sequence_end": index * 12,
        "quality_valid_market_count": 12,
        "positive_guard_compatible_trade_lcb_row_count": positive,
        "guard_accepted_unique_market_count": len(accepted_ids),
        "guard_accepted_market_ids": accepted_ids,
        "guard_accepted_market_ids_by_side": {
            "UP": list(accepted.get("UP") or []),
            "DOWN": list(accepted.get("DOWN") or []),
        },
        "guard_accepted_by_side": {
            "UP": len(accepted.get("UP") or []),
            "DOWN": len(accepted.get("DOWN") or []),
        },
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
