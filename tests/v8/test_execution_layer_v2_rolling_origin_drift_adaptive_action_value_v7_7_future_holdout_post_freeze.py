from __future__ import annotations

import json
from pathlib import Path

import pytest

import bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_post_freeze as post_freeze
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout import (
    EXACT_MARKET_COUNT,
    FROZEN_PLAN_SHA256,
    SCHEMA_PREFIX,
    _safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout_post_freeze import (
    V77FutureEvaluationConfig,
    V77FutureSettlementConfig,
    build_v7_7_future_settled_index,
    evaluate_v7_7_future_pnl_gate,
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
RUNTIME_PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
)
RUNTIME_PROFILE_SHA256 = "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _freeze_fixture(tmp_path: Path) -> tuple[Path, str, list[dict], list[dict]]:
    selected = [
        {
            "market_id": f"market-{index:03d}",
            "run_id": f"round-{index:03d}",
            "market_end_ts": 100 + index,
        }
        for index in range(EXACT_MARKET_COUNT)
    ]
    candidate = [
        {
            "market_id": f"market-{index:03d}",
            "decision_ts": 10 + index,
            "max_input_ts": 9 + index,
            "market_close_ts": 100 + index,
            "side": "UP",
            "action": "BUY_UP_SELL_BEFORE_CLOSE",
            "microstructure_snapshot": {"time_to_close_seconds": 90.0},
            "source_score_mutated": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
            **_safety_fields(),
        }
        for index in range(40)
    ]
    baseline = [dict(row) for row in candidate]
    selected_path = tmp_path / "selected.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(candidate_path, candidate)
    _write_jsonl(baseline_path, baseline)
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1",
        "run_id": "freeze",
        "plan": _descriptor(PLAN_PATH),
        "selected_rows": _descriptor(selected_path),
        "v7_7_runtime": _descriptor(candidate_path),
        "v6_7_runtime": _descriptor(baseline_path),
        "exact_market_count": EXACT_MARKET_COUNT,
        "decision_freeze_created_ts": 150,
        "decision_freeze_written_before_target_access": True,
        "target_free_freeze_passed": True,
        "future_target_access_allowed": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_or_model_tuning_performed": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = "a" * 64
    manifest_path = tmp_path / "freeze.json"
    _write_json(manifest_path, manifest)
    return manifest_path, _sha256_file(manifest_path), candidate, baseline


def _settled_entry_fixture(tmp_path: Path, market_id: str) -> dict:
    feature_path = tmp_path / "settled_features.jsonl"
    label_path = tmp_path / "settled_labels.jsonl"
    resolution_path = tmp_path / "settled_resolutions.jsonl"
    for path in (feature_path, label_path, resolution_path):
        if not path.exists():
            _write_jsonl(path, [])
    return {
        "market_id": market_id,
        "official_read_only_resolution": True,
        "source_outcome_blind_round_mutated": False,
        "feature_rows": _descriptor(feature_path),
        "label_rows": _descriptor(label_path),
        "resolution_events": _descriptor(resolution_path),
    }


def _successful_settlement(tmp_path: Path, market_id: str) -> dict:
    return {
        "market_id": market_id,
        "settled_corpus_ready": True,
        "index_entry": _settled_entry_fixture(tmp_path, market_id),
    }


def test_read_only_settlement_emits_exact_120_index_and_blocks_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path, freeze_sha, _, _ = _freeze_fixture(tmp_path)

    def fake_finalize(selected_rows: list[dict], **_: object) -> list[dict]:
        return [
            _successful_settlement(tmp_path, str(row["market_id"]))
            for row in selected_rows
        ]

    monkeypatch.setattr(post_freeze, "_finalize_selected_rounds", fake_finalize)
    result = build_v7_7_future_settled_index(
        V77FutureSettlementConfig(
            run_id="settled",
            output_dir=tmp_path / "runs",
            target_free_freeze_manifest_path=freeze_path,
            expected_target_free_freeze_manifest_sha256=freeze_sha,
            implementation_commit="b" * 40,
            target_access_started_ts=300,
        ),
        clock_ms_fn=lambda: 350,
    )

    assert result["report"]["settled_index_ready"] is True
    assert result["index"]["entry_count"] == EXACT_MARKET_COUNT
    assert result["index"]["source_outcome_blind_rounds_mutated"] is False
    assert result["index"]["official_read_only_resolution"] is True
    assert result["index"]["v8_execution_handoff_allowed"] is False

    with pytest.raises(ValueError, match="already been consumed"):
        build_v7_7_future_settled_index(
            V77FutureSettlementConfig(
                run_id="settled-again",
                output_dir=tmp_path / "runs",
                target_free_freeze_manifest_path=freeze_path,
                expected_target_free_freeze_manifest_sha256=freeze_sha,
                implementation_commit="b" * 40,
                target_access_started_ts=301,
            ),
            clock_ms_fn=lambda: 351,
        )


def test_unresolved_settlement_remains_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path, freeze_sha, _, _ = _freeze_fixture(tmp_path)

    def fake_finalize(selected_rows: list[dict], **_: object) -> list[dict]:
        return [
            {
                "market_id": str(row["market_id"]),
                "settled_corpus_ready": False,
                "failure": {
                    "market_id": str(row["market_id"]),
                    "reason_codes": ["official_resolution_still_pending"],
                    "pending_resolution": True,
                },
            }
            for row in selected_rows
        ]

    monotonic = iter([0.0, 11.0])
    monkeypatch.setattr(post_freeze, "_finalize_selected_rounds", fake_finalize)
    result = build_v7_7_future_settled_index(
        V77FutureSettlementConfig(
            run_id="unresolved",
            output_dir=tmp_path / "runs",
            target_free_freeze_manifest_path=freeze_path,
            expected_target_free_freeze_manifest_sha256=freeze_sha,
            implementation_commit="c" * 40,
            target_access_started_ts=300,
            settlement_max_wait_seconds=10,
        ),
        monotonic_fn=lambda: next(monotonic),
        sleep_fn=lambda _: None,
        clock_ms_fn=lambda: 350,
    )

    assert result["report"]["settled_index_ready"] is False
    assert result["index"] is None
    assert result["report"]["unresolved_or_failed_market_count"] == EXACT_MARKET_COUNT
    assert result["report"]["blocking_reason_codes"] == ["settled_window_incomplete"]
    assert result["report"]["promotion_evidence_eligible"] is False


def test_equal_positive_future_pnl_passes_without_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_path, freeze_sha, _, _ = _freeze_fixture(tmp_path)

    def fake_finalize(selected_rows: list[dict], **_: object) -> list[dict]:
        return [
            _successful_settlement(tmp_path, str(row["market_id"]))
            for row in selected_rows
        ]

    monkeypatch.setattr(post_freeze, "_finalize_selected_rounds", fake_finalize)
    settled = build_v7_7_future_settled_index(
        V77FutureSettlementConfig(
            run_id="settled",
            output_dir=tmp_path / "settlement-runs",
            target_free_freeze_manifest_path=freeze_path,
            expected_target_free_freeze_manifest_sha256=freeze_sha,
            implementation_commit="d" * 40,
            target_access_started_ts=300,
        ),
        clock_ms_fn=lambda: 350,
    )

    def fake_runtime_targets(
        decisions: list[dict], *, role: str, **_: object
    ) -> list[dict]:
        assert role.startswith("future_unseen_holdout")
        return [
            {
                "market_id": row["market_id"],
                "decision_ts": row["decision_ts"],
                "max_input_ts": row["max_input_ts"],
                "side": row["side"],
                "action": row["action"],
                "runtime_policy_after_cost_net_pnl_at_frozen_size": 0.01,
                "target_available_only_post_exit_or_official_resolution": True,
                "target_used_as_decision_time_input": False,
            }
            for row in decisions
        ]

    monkeypatch.setattr(post_freeze, "_runtime_targets_for_decisions", fake_runtime_targets)
    result_path = settled["index_path"]
    result = evaluate_v7_7_future_pnl_gate(
        V77FutureEvaluationConfig(
            run_id="evaluate",
            output_dir=tmp_path / "evaluation-runs",
            target_free_freeze_manifest_path=freeze_path,
            expected_target_free_freeze_manifest_sha256=freeze_sha,
            settled_index_path=result_path,
            expected_settled_index_sha256=_sha256_file(result_path),
            runtime_policy_profile_path=RUNTIME_PROFILE_PATH,
            expected_runtime_policy_profile_sha256=RUNTIME_PROFILE_SHA256,
            implementation_commit="e" * 40,
            evaluation_started_ts=400,
        )
    )

    report = result["report"]
    assert report["candidate_after_cost_pnl"] == pytest.approx(0.4)
    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["equality_passes_noninferiority"] is True
    assert report["future_pnl_gate_passed"] is True
    assert report["model_improvement_demonstrated"] is False
    assert report["paper_candidate_allowed"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["promotion_evidence_eligible"] is False


def test_plan_and_runtime_pins_remain_frozen() -> None:
    assert _sha256_file(PLAN_PATH) == FROZEN_PLAN_SHA256
    assert _sha256_file(RUNTIME_PROFILE_PATH) == RUNTIME_PROFILE_SHA256
