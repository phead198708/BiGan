from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    build_regime_emergent_target_free_support,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_confirmatory import (
    FROZEN_EVALUATION_PROFILE_SHA256,
    FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
    V68ConfirmatoryPostFreezeConfig,
    _validate_freeze,
    run_v6_8_confirmatory_post_freeze,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_confirmatory import (
    SCHEMA_PREFIX as POST_FREEZE_SCHEMA_PREFIX,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_pipeline import (
    SCHEMA_PREFIX,
    select_v6_8_confirmatory_index_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_regime_emergent_pnl_v6_8_evaluation_v1.json"
)
RUNTIME_PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _index_row(sequence: int, *, start_ts: int, valid: bool = True) -> dict:
    return {
        "sequence": sequence,
        "market_id": f"future-market-{sequence:03d}",
        "scheduled_round_start_ts": start_ts,
        "market_start_ts": start_ts,
        "market_end_ts": start_ts + 300_000,
        "capture_quality_valid": valid,
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
        **_blocked_safety_fields(),
    }


def _calibration_adoption(tmp_path: Path) -> dict:
    selected = [
        {
            "sequence": index + 1,
            "market_id": f"calibration-{index:03d}",
            "market_end_ts": 2_000_000 + index,
        }
        for index in range(60)
    ]
    attempted = [
        {
            "sequence": index + 1,
            "market_id": f"attempted-{index:03d}",
        }
        for index in range(66)
    ]
    selected_path = tmp_path / "calibration-selected.jsonl"
    attempted_path = tmp_path / "calibration-attempted.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(attempted_path, attempted)
    return {
        "schema_version": (
            "bigan-v8-regime-emergent-pnl-v6-8-sealed-decision-adoption-manifest-v1"
        ),
        "role": "fresh_calibration",
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "side_count_hard_gate_enabled": False,
        "selected_window_rows": _descriptor(selected_path),
        "attempted_window_rows": _descriptor(attempted_path),
    }


def _decision(index: int) -> dict:
    decision_ts = 4_100_000 + index
    return {
        "market_id": f"confirmatory-{index:03d}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "side": "UP",
        "action": "BUY_UP_SELL_BEFORE_CLOSE",
        "v6_8_calibrated_runtime_pnl_lcb": 0.05,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
    }


def test_confirmatory_support_separates_window_size_from_total_bet_support() -> None:
    rows = [_decision(index) for index in range(40)]
    support = build_regime_emergent_target_free_support(
        rows,
        exact_window_market_count=120,
        expected_window_market_count=120,
        required_total_market_count=40,
        score_field="v6_8_calibrated_runtime_pnl_lcb",
    )

    assert support["target_free_support_gate_passed"] is True
    assert support["count_by_side"] == {"DOWN": 0, "UP": 40}
    assert support["minimum_per_side_required"] is None
    assert support["side_count_hard_gate_enabled"] is False


def test_confirmatory_window_is_strictly_later_and_skips_boundary_row(
    tmp_path: Path,
) -> None:
    adoption = _calibration_adoption(tmp_path)
    boundary = 2_000_059
    rows = [
        _index_row(67, start_ts=boundary),
        *[_index_row(68 + index, start_ts=boundary + 1 + index * 300_000) for index in range(120)],
    ]

    selected, attempted = select_v6_8_confirmatory_index_rows(
        rows,
        calibration_adoption_manifest=adoption,
    )

    assert len(selected) == 120
    assert len(attempted) == 120
    assert selected[0]["sequence"] == 68
    assert all(row["scheduled_round_start_ts"] > boundary for row in selected)


def test_confirmatory_freeze_accepts_one_sided_regime_and_keeps_safety_blocked(
    tmp_path: Path,
) -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    calibration_adoption = _calibration_adoption(tmp_path)
    adoption_path = tmp_path / "calibration-adoption.json"
    _write_json(adoption_path, calibration_adoption)
    boundary = max(
        row["market_end_ts"]
        for row in [
            json.loads(line)
            for line in Path(calibration_adoption["selected_window_rows"]["path"])
            .read_text()
            .splitlines()
        ]
    )
    selected = [
        {
            "market_id": f"confirmatory-{index:03d}",
            "scheduled_round_start_ts": boundary + 1 + index * 300_000,
            "market_end_ts": boundary + 300_001 + index * 300_000,
        }
        for index in range(120)
    ]
    decisions = [_decision(index) for index in range(40)]
    selected_path = tmp_path / "confirmatory-selected.jsonl"
    decisions_path = tmp_path / "confirmatory-decisions.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=120,
        expected_window_market_count=120,
        required_total_market_count=40,
        score_field="v6_8_calibrated_runtime_pnl_lcb",
    )
    decision_path = tmp_path / "decision-freeze.json"
    _write_json(
        decision_path,
        {
            "selected_window_market_ids": [row["market_id"] for row in selected],
            "regime_emergent_target_free_support": support,
            "future_target_access_allowed": True,
        },
    )
    calibration_path = tmp_path / "calibration.json"
    _write_json(
        calibration_path,
        {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "calibration_gate_passed": True,
            "calibration_gate_blocking_reason_codes": [],
            "side_count_hard_gate_enabled": False,
        },
    )
    freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "role": "future_confirmatory",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "future_target_access_allowed": True,
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "selected_window_rows": _descriptor(selected_path),
        "v6_8_selected_decisions": _descriptor(decisions_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "calibration_adoption_manifest": _descriptor(adoption_path),
        "calibration_artifact": _descriptor(calibration_path),
        **_blocked_safety_fields(),
    }

    _validate_freeze(freeze, profile=profile, profile_path=PROFILE_PATH)
    assert freeze["promotion_evidence_eligible"] is False
    assert freeze["#134_resume_allowed"] is False
    assert freeze["#146_start_allowed"] is False

    decisions[0]["market_id"] = "outside-frozen-window"
    _write_jsonl(decisions_path, decisions)
    freeze["v6_8_selected_decisions"] = _descriptor(decisions_path)
    with pytest.raises(ValueError, match="decision-freeze evidence"):
        _validate_freeze(freeze, profile=profile, profile_path=PROFILE_PATH)


def test_confirmatory_evaluation_config_requires_settled_target_inputs() -> None:
    with pytest.raises(ValueError, match="evaluation input missing"):
        V68ConfirmatoryPostFreezeConfig(
            stage="evaluate_confirmatory",
            run_id="missing-targets",
            output_dir="runs",
            evaluation_profile_path="profile.json",
            expected_evaluation_profile_sha256="1" * 64,
            prediction_freeze_manifest_path="freeze.json",
            expected_prediction_freeze_manifest_sha256="2" * 64,
            implementation_commit="3" * 40,
            stage_started_ts=1,
        )


def test_confirmatory_evaluation_is_single_use_and_side_agnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adoption = _calibration_adoption(tmp_path)
    adoption_path = tmp_path / "adoption.json"
    _write_json(adoption_path, adoption)
    boundary = max(
        json.loads(line)["market_end_ts"]
        for line in Path(adoption["selected_window_rows"]["path"]).read_text().splitlines()
    )
    selected = [
        {
            "market_id": f"confirmatory-{index:03d}",
            "scheduled_round_start_ts": boundary + 1 + index * 300_000,
            "market_end_ts": boundary + 300_001 + index * 300_000,
        }
        for index in range(120)
    ]
    decisions = [_decision(index) for index in range(40)]
    predictions = [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "max_input_ts": row["max_input_ts"],
            "action": row["action"],
            "microstructure_snapshot": {"time_to_close_seconds": 200.0},
        }
        for row in decisions
    ]
    legacy = [
        {
            "market_id": row["market_id"],
            "decision_ts": row["decision_ts"],
            "selected_side": row["side"],
            "executed_action": row["action"],
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "execution_guard_order_allowed": True,
        }
        for row in decisions
    ]
    selected_path = tmp_path / "selected.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    legacy_path = tmp_path / "legacy.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(legacy_path, legacy)
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=120,
        expected_window_market_count=120,
        required_total_market_count=40,
        score_field="v6_8_calibrated_runtime_pnl_lcb",
    )
    decision_path = tmp_path / "decision.json"
    _write_json(
        decision_path,
        {
            "decision_freeze_created_ts": 4_000_000,
            "selected_window_market_ids": [row["market_id"] for row in selected],
            "regime_emergent_target_free_support": support,
            "future_target_access_allowed": True,
        },
    )
    calibration_path = tmp_path / "calibration.json"
    _write_json(
        calibration_path,
        {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "calibration_gate_passed": True,
            "calibration_gate_blocking_reason_codes": [],
            "side_count_hard_gate_enabled": False,
        },
    )
    freeze = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "role": "future_confirmatory",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "future_target_access_allowed": True,
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "selected_window_rows": _descriptor(selected_path),
        "v6_8_selected_decisions": _descriptor(decisions_path),
        "v6_2_target_free_predictions": _descriptor(predictions_path),
        "matched_legacy_guard_replay": _descriptor(legacy_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "calibration_adoption_manifest": _descriptor(adoption_path),
        "calibration_artifact": _descriptor(calibration_path),
        **_blocked_safety_fields(),
    }
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)
    evidence_path = tmp_path / "evidence.jsonl"
    _write_jsonl(evidence_path, [])
    entries = [
        {
            "market_id": row["market_id"],
            "feature_rows": _descriptor(evidence_path),
            "label_rows": _descriptor(evidence_path),
            "resolution_events": _descriptor(evidence_path),
            "official_read_only_resolution": True,
            "source_outcome_blind_round_mutated": False,
        }
        for row in selected
    ]
    index = {
        "schema_version": f"{POST_FREEZE_SCHEMA_PREFIX}-settled-corpus-index-v1",
        "role": "future_confirmatory",
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "decision_freeze_sha256": freeze["accepted_bet_decision_freeze"]["sha256"],
        "entry_count": 120,
        "entries": entries,
        "index_finalized_ts": 5_000_000,
        "outcomes_used_for_decision_selection_or_tuning": False,
        "side_quota_applied": False,
        **_blocked_safety_fields(),
    }
    index_path = tmp_path / "settled-index.json"
    _write_json(index_path, index)

    def fake_targets(rows, *, run_id, **_kwargs):
        pnl = -0.01 if run_id.endswith("-legacy") else 0.02
        return [
            {
                **row,
                "runtime_policy_after_cost_net_pnl_per_contract": pnl,
                "runtime_policy_after_cost_net_pnl_at_frozen_size": pnl,
                "target_available_only_post_exit_or_official_resolution": True,
                "target_used_as_decision_time_input": False,
            }
            for row in rows
        ]

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "execution_layer_v2_regime_emergent_pnl_v6_8_confirmatory."
        "_runtime_targets_for_decisions",
        fake_targets,
    )
    config = V68ConfirmatoryPostFreezeConfig(
        stage="evaluate_confirmatory",
        run_id="confirmatory-evaluation",
        output_dir=tmp_path / "runs",
        evaluation_profile_path=PROFILE_PATH,
        expected_evaluation_profile_sha256=FROZEN_EVALUATION_PROFILE_SHA256,
        prediction_freeze_manifest_path=freeze_path,
        expected_prediction_freeze_manifest_sha256=_descriptor(freeze_path)["sha256"],
        runtime_policy_profile_path=RUNTIME_PROFILE_PATH,
        expected_runtime_policy_profile_sha256=FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
        settled_corpus_index_path=index_path,
        expected_settled_corpus_index_sha256=_descriptor(index_path)["sha256"],
        implementation_commit="7" * 40,
        stage_started_ts=6_000_000,
    )

    result = run_v6_8_confirmatory_post_freeze(config)

    assert result["report"]["confirmatory_execution_pnl_gate_passed"] is True
    assert result["report"]["accepted_side_distribution_diagnostic"] == {"UP": 40}
    assert result["report"]["side_count_hard_gate_enabled"] is False
    assert result["manifest"]["promotion_evidence_eligible"] is False
    with pytest.raises(ValueError, match="already consumed"):
        run_v6_8_confirmatory_post_freeze(replace(config, run_id="confirmatory-evaluation-rerun"))
