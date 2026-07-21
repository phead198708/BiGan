from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    _target_free_support,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    FROZEN_EVALUATION_PROFILE_SHA256,
    FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
    SCHEMA_PREFIX,
    V67PostFreezeConfig,
    _legacy_guard_accepted_sbc_decisions,
    _runtime_targets_for_decisions,
    _single_use_claim_path,
    _validate_settled_index,
    _validate_window_freeze,
    _write_single_use_claim,
    run_v6_7_post_freeze,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_p_up_semantic_compatibility_v6_7_evaluation_v1.json"
)
RUNTIME_PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
)


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _decision(index: int) -> dict:
    side = "UP" if index < 20 else "DOWN"
    return {
        "market_id": f"market-{index:03d}",
        "decision_ts": 2_000_000 + index,
        "market_close_ts": 2_300_000 + index,
        "max_input_ts": 1_999_000 + index,
        "side": side,
        "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
        "v6_7_base_score": 0.20,
        "microstructure_snapshot": {"time_to_close_seconds": 240.0},
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }


def _freeze_fixture(tmp_path: Path) -> tuple[dict, Path, list[dict], list[dict]]:
    profile = _profile()
    selected = [
        {
            "market_id": f"market-{index:03d}",
            "market_start_ts": 1_900_000 + index,
            "market_end_ts": 2_300_000 + index,
        }
        for index in range(60)
    ]
    decisions = [_decision(index) for index in range(60)]
    selected_path = tmp_path / "selected.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    support = _target_free_support(
        decisions,
        profile=profile,
        role="fresh_calibration",
        exact_window_market_count=60,
    )
    decision_artifact = {
        "role": "fresh_calibration",
        "decision_freeze_created_ts": 3_000_000,
        "selected_window_market_ids": [row["market_id"] for row in selected],
        "target_free_support": support,
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        **_blocked_safety_fields(),
    }
    decision_path = tmp_path / "decision.json"
    _write_json(decision_path, decision_artifact)
    manifest = {
        "schema_version": (
            "bigan-v8-p-up-semantic-execution-compatibility-v6-7-window-freeze-manifest-v1"
        ),
        "role": "fresh_calibration",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "selected_window_rows": _descriptor(selected_path),
        "v6_7_selected_decisions": _descriptor(decisions_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        **_blocked_safety_fields(),
    }
    return manifest, decisions_path, selected, decisions


@pytest.mark.parametrize(
    ("stage", "role"),
    [
        ("settle", "fresh_calibration"),
        ("settle", "future_confirmatory"),
        ("calibrate", "fresh_calibration"),
        ("evaluate_confirmatory", "future_confirmatory"),
    ],
)
def test_post_freeze_stage_config_is_explicit(stage: str, role: str) -> None:
    extra = {}
    if stage != "settle":
        extra = {
            "runtime_policy_profile_path": "runtime.json",
            "expected_runtime_policy_profile_sha256": "3" * 64,
            "settled_corpus_index_path": "index.json",
            "expected_settled_corpus_index_sha256": "4" * 64,
        }
    config = V67PostFreezeConfig(
        stage=stage,
        role=role,
        run_id="stage-test",
        output_dir="runs",
        evaluation_profile_path="profile.json",
        expected_evaluation_profile_sha256="1" * 64,
        prediction_freeze_manifest_path="freeze.json",
        expected_prediction_freeze_manifest_sha256="2" * 64,
        implementation_commit="5" * 40,
        stage_started_ts=1,
        **extra,
    )

    assert config.stage == stage
    assert config.role == role


def test_window_freeze_validation_rejects_target_leakage(tmp_path: Path) -> None:
    manifest, decisions_path, _, decisions = _freeze_fixture(tmp_path)
    _validate_window_freeze(
        manifest,
        role="fresh_calibration",
        profile=_profile(),
        profile_path=PROFILE_PATH,
    )

    decisions[0]["settlement_pnl"] = 1.0
    _write_jsonl(decisions_path, decisions)
    manifest["v6_7_selected_decisions"] = _descriptor(decisions_path)
    with pytest.raises(ValueError, match="target evidence"):
        _validate_window_freeze(
            manifest,
            role="fresh_calibration",
            profile=_profile(),
            profile_path=PROFILE_PATH,
        )


def test_settled_index_requires_exact_quarantined_window(tmp_path: Path) -> None:
    freeze, _, selected, _ = _freeze_fixture(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, freeze)
    evidence_path = tmp_path / "evidence.jsonl"
    _write_jsonl(evidence_path, [{"evidence": True}])
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
        "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
        "role": "fresh_calibration",
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "decision_freeze_sha256": freeze["accepted_bet_decision_freeze"]["sha256"],
        "entry_count": 60,
        "entries": entries,
        "index_finalized_ts": 4_000_000,
        "outcomes_used_for_decision_selection_or_tuning": False,
        **_blocked_safety_fields(),
    }
    _validate_settled_index(
        index,
        freeze=freeze,
        freeze_path=freeze_path,
        role="fresh_calibration",
        evaluation_started_ts=5_000_000,
    )

    index["entries"] = entries[:-1]
    with pytest.raises(ValueError, match="not evaluation eligible"):
        _validate_settled_index(
            index,
            freeze=freeze,
            freeze_path=freeze_path,
            role="fresh_calibration",
            evaluation_started_ts=5_000_000,
        )


def test_legacy_mapper_uses_frozen_prediction_identity() -> None:
    action = "BUY_DOWN_SELL_BEFORE_CLOSE"
    predictions = [
        {
            "market_id": "market-a",
            "decision_ts": 2_000,
            "max_input_ts": 1_999,
            "action": action,
            "microstructure_snapshot": {"time_to_close_seconds": 180.0},
        }
    ]
    replay = [
        {
            "market_id": "market-a",
            "decision_ts": 2_000,
            "selected_side": "DOWN",
            "executed_action": action,
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "execution_guard_order_allowed": True,
        },
        {
            "market_id": "blocked",
            "decision_ts": 2_001,
            "selected_side": "UP",
            "executed_action": "BUY_UP_SELL_BEFORE_CLOSE",
            "selected_action_family": "SELL_BEFORE_CLOSE",
            "execution_guard_order_allowed": False,
        },
    ]

    rows = _legacy_guard_accepted_sbc_decisions(replay, predictions=predictions)

    assert rows == [
        {
            "market_id": "market-a",
            "decision_ts": 2_000,
            "max_input_ts": 1_999,
            "side": "DOWN",
            "action": action,
            "microstructure_snapshot": {"time_to_close_seconds": 180.0},
        }
    ]


def test_runtime_target_builder_filters_exact_decision_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_path = tmp_path / "features.jsonl"
    label_path = tmp_path / "labels.jsonl"
    resolution_path = tmp_path / "resolution.jsonl"
    _write_jsonl(feature_path, [{"decision_ts": 2_000}])
    _write_jsonl(
        label_path,
        [
            {"decision_ts": 2_000, "action": "BUY_UP_SELL_BEFORE_CLOSE"},
            {"decision_ts": 2_001, "action": "BUY_DOWN_SELL_BEFORE_CLOSE"},
        ],
    )
    _write_jsonl(resolution_path, [{"outcome": "UP"}])
    seen: dict[str, object] = {}

    def fake_target_builder(**kwargs):
        seen.update(kwargs)
        decision = kwargs["decision_rows"][0]
        return (
            [
                {
                    **decision,
                    "runtime_policy_after_cost_net_pnl_per_contract": 0.1,
                    "runtime_policy_after_cost_net_pnl_at_frozen_size": 0.02,
                    "position_lifecycle_class": "closed_before_settlement",
                    "resolved_outcome": "UP",
                    "target_available_only_post_exit_or_official_resolution": True,
                    "target_used_as_decision_time_input": False,
                }
            ],
            {},
        )

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze."
        "_market_runtime_target_rows",
        fake_target_builder,
    )
    decision = _decision(0)
    decision["market_id"] = "market-a"
    decision["decision_ts"] = 2_000
    decision["max_input_ts"] = 1_999
    rows = _runtime_targets_for_decisions(
        [decision],
        settled_entries=[
            {
                "market_id": "market-a",
                "run_id": "round-a",
                "feature_rows": _descriptor(feature_path),
                "label_rows": _descriptor(label_path),
                "resolution_events": _descriptor(resolution_path),
            }
        ],
        runtime_profile={},
        run_id="target-test",
        role="fresh_calibration",
    )

    assert len(rows) == 1
    assert seen["label_rows"] == [{"decision_ts": 2_000, "action": "BUY_UP_SELL_BEFORE_CLOSE"}]
    assert seen["decision_rows"][0]["max_input_ts"] == 1_999


def test_target_window_single_use_claim_fails_closed(tmp_path: Path) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}\n")
    claim_path = _single_use_claim_path(freeze_path, "fresh_calibration")
    claim = {"claim_id": "one"}

    _write_single_use_claim(claim_path, claim)

    with pytest.raises(ValueError, match="already been consumed"):
        _write_single_use_claim(claim_path, {"claim_id": "two"})
    assert json.loads(claim_path.read_text()) == claim


def _settled_index_fixture(
    tmp_path: Path,
    *,
    freeze: dict,
    freeze_path: Path,
    selected: list[dict],
    role: str,
) -> tuple[Path, Path]:
    features_path = tmp_path / f"{role}-features.jsonl"
    labels_path = tmp_path / f"{role}-labels.jsonl"
    resolution_path = tmp_path / f"{role}-resolution.jsonl"
    _write_jsonl(
        features_path,
        [
            {"market_id": row["market_id"], "decision_ts": 2_000_000 + index}
            for index, row in enumerate(selected)
        ],
    )
    _write_jsonl(labels_path, [])
    _write_jsonl(resolution_path, [{"resolved_outcome": "UP"}])
    entries = [
        {
            "market_id": row["market_id"],
            "run_id": f"round-{index:03d}",
            "feature_rows": _descriptor(features_path),
            "label_rows": _descriptor(labels_path),
            "resolution_events": _descriptor(resolution_path),
            "official_read_only_resolution": True,
            "source_outcome_blind_round_mutated": False,
        }
        for index, row in enumerate(selected)
    ]
    index = {
        "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
        "role": role,
        "prediction_freeze_manifest": _descriptor(freeze_path),
        "decision_freeze_sha256": freeze["accepted_bet_decision_freeze"]["sha256"],
        "entry_count": len(entries),
        "entries": entries,
        "index_finalized_ts": 4_000_000,
        "outcomes_used_for_decision_selection_or_tuning": False,
        **_blocked_safety_fields(),
    }
    index_path = tmp_path / f"{role}-settled-index.json"
    _write_json(index_path, index)
    return index_path, labels_path


def _fake_runtime_target_builder(*, runtime_pnl: float):
    def build(**kwargs):
        decision = kwargs["decision_rows"][0]
        size = 0.2
        return (
            [
                {
                    **decision,
                    "runtime_policy_after_cost_net_pnl_per_contract": runtime_pnl,
                    "runtime_policy_after_cost_net_pnl_at_frozen_size": (runtime_pnl * size),
                    "position_lifecycle_class": "closed_before_settlement",
                    "resolved_outcome": "UP",
                    "target_available_only_post_exit_or_official_resolution": True,
                    "target_used_as_decision_time_input": False,
                }
            ],
            {},
        )

    return build


def test_full_calibration_stage_freezes_artifact_and_consumes_window_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, _, selected, _ = _freeze_fixture(tmp_path)
    freeze_path = tmp_path / "calibration-freeze.json"
    _write_json(freeze_path, freeze)
    index_path, _ = _settled_index_fixture(
        tmp_path,
        freeze=freeze,
        freeze_path=freeze_path,
        selected=selected,
        role="fresh_calibration",
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze."
        "_market_runtime_target_rows",
        _fake_runtime_target_builder(runtime_pnl=0.15),
    )
    config = V67PostFreezeConfig(
        stage="calibrate",
        role="fresh_calibration",
        run_id="calibration-stage-integration",
        output_dir=tmp_path / "runs",
        evaluation_profile_path=PROFILE_PATH,
        expected_evaluation_profile_sha256=FROZEN_EVALUATION_PROFILE_SHA256,
        prediction_freeze_manifest_path=freeze_path,
        expected_prediction_freeze_manifest_sha256=_descriptor(freeze_path)["sha256"],
        runtime_policy_profile_path=RUNTIME_PROFILE_PATH,
        expected_runtime_policy_profile_sha256=FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
        settled_corpus_index_path=index_path,
        expected_settled_corpus_index_sha256=_descriptor(index_path)["sha256"],
        implementation_commit="6" * 40,
        stage_started_ts=5_000_000,
    )

    result = run_v6_7_post_freeze(config)

    assert result["report"]["calibration_gate_passed"] is True
    assert result["manifest"]["future_confirmatory_freeze_allowed"] is True
    assert result["manifest"]["source_model_candidate_eligible"] is False
    with pytest.raises(ValueError, match="already been consumed"):
        run_v6_7_post_freeze(replace(config, run_id="calibration-stage-rerun"))


def _confirmatory_freeze_fixture(
    tmp_path: Path,
) -> tuple[dict, Path, list[dict]]:
    profile = _profile()
    selected = [
        {
            "market_id": f"future-{index:03d}",
            "market_start_ts": 5_000_000 + index,
            "market_end_ts": 5_300_000 + index,
        }
        for index in range(120)
    ]
    decisions = []
    predictions = []
    legacy = []
    for index, selected_row in enumerate(selected):
        side = "UP" if index < 60 else "DOWN"
        action = f"BUY_{side}_SELL_BEFORE_CLOSE"
        row = {
            "market_id": selected_row["market_id"],
            "decision_ts": 5_100_000 + index,
            "market_close_ts": selected_row["market_end_ts"],
            "max_input_ts": 5_099_000 + index,
            "side": side,
            "action": action,
            "v6_7_base_score": 0.20,
            "v6_7_calibrated_runtime_pnl_lcb": 0.10,
            "microstructure_snapshot": {"time_to_close_seconds": 200.0},
            "labels_outcomes_resolution_or_pnl_opened": False,
            "source_score_mutated": False,
            **_blocked_safety_fields(),
        }
        decisions.append(row)
        predictions.append(
            {
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "max_input_ts": row["max_input_ts"],
                "action": action,
                "microstructure_snapshot": row["microstructure_snapshot"],
            }
        )
        legacy.append(
            {
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "selected_side": side,
                "executed_action": action,
                "selected_action_family": "SELL_BEFORE_CLOSE",
                "execution_guard_order_allowed": True,
            }
        )
    selected_path = tmp_path / "confirmatory-selected.jsonl"
    decisions_path = tmp_path / "confirmatory-decisions.jsonl"
    predictions_path = tmp_path / "confirmatory-predictions.jsonl"
    legacy_path = tmp_path / "confirmatory-legacy.jsonl"
    calibration_path = tmp_path / "calibration-artifact.json"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(legacy_path, legacy)
    _write_json(
        calibration_path,
        {
            "calibration_gate_passed": True,
            "calibration_gate_blocking_reason_codes": [],
        },
    )
    support = _target_free_support(
        decisions,
        profile=profile,
        role="future_confirmatory",
        exact_window_market_count=120,
    )
    decision_path = tmp_path / "confirmatory-decision-freeze.json"
    _write_json(
        decision_path,
        {
            "role": "future_confirmatory",
            "decision_freeze_created_ts": 6_000_000,
            "selected_window_market_ids": [row["market_id"] for row in selected],
            "target_free_support": support,
            "future_target_access_allowed": True,
            "labels_outcomes_resolution_or_pnl_opened": False,
            "settlement_provider_called": False,
            **_blocked_safety_fields(),
        },
    )
    freeze = {
        "schema_version": (
            "bigan-v8-p-up-semantic-execution-compatibility-v6-7-window-freeze-manifest-v1"
        ),
        "role": "future_confirmatory",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "selected_window_rows": _descriptor(selected_path),
        "v6_7_selected_decisions": _descriptor(decisions_path),
        "v6_2_target_free_predictions": _descriptor(predictions_path),
        "matched_legacy_guard_replay": _descriptor(legacy_path),
        "calibration_artifact": _descriptor(calibration_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        **_blocked_safety_fields(),
    }
    freeze_path = tmp_path / "confirmatory-freeze.json"
    _write_json(freeze_path, freeze)
    return freeze, freeze_path, selected


def test_full_confirmatory_stage_keeps_pass_diagnostic_and_unlocks_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze, freeze_path, selected = _confirmatory_freeze_fixture(tmp_path)
    index_path, _ = _settled_index_fixture(
        tmp_path,
        freeze=freeze,
        freeze_path=freeze_path,
        selected=selected,
        role="future_confirmatory",
    )

    def fake_target_builder(**kwargs):
        pnl = -0.01 if kwargs["run_id"].endswith("-legacy") else 0.02
        return _fake_runtime_target_builder(runtime_pnl=pnl)(**kwargs)

    monkeypatch.setattr(
        "bigan.v8.polymarket.training."
        "execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze."
        "_market_runtime_target_rows",
        fake_target_builder,
    )
    result = run_v6_7_post_freeze(
        V67PostFreezeConfig(
            stage="evaluate_confirmatory",
            role="future_confirmatory",
            run_id="confirmatory-stage-integration",
            output_dir=tmp_path / "runs",
            evaluation_profile_path=PROFILE_PATH,
            expected_evaluation_profile_sha256=FROZEN_EVALUATION_PROFILE_SHA256,
            prediction_freeze_manifest_path=freeze_path,
            expected_prediction_freeze_manifest_sha256=_descriptor(freeze_path)["sha256"],
            runtime_policy_profile_path=RUNTIME_PROFILE_PATH,
            expected_runtime_policy_profile_sha256=(FROZEN_RUNTIME_POLICY_PROFILE_SHA256),
            settled_corpus_index_path=index_path,
            expected_settled_corpus_index_sha256=_descriptor(index_path)["sha256"],
            implementation_commit="7" * 40,
            stage_started_ts=7_000_000,
        )
    )

    assert result["report"]["confirmatory_side_only_pnl_gate_passed"] is True
    assert result["report"]["candidate_after_cost_pnl"] > 0.0
    assert result["report"]["matched_legacy_after_cost_pnl"] < 0.0
    assert result["manifest"]["source_model_candidate_eligible"] is False
    assert result["manifest"]["promotion_evidence_eligible"] is False
    assert result["manifest"]["#134_resume_allowed"] is False
    assert result["manifest"]["#146_start_allowed"] is False
