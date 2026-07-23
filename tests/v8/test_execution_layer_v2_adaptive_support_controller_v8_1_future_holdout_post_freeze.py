from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_post_freeze as post,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_plan.json"
)
RUNTIME_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def _freeze(tmp_path: Path) -> tuple[Path, list[str]]:
    market_ids = [f"market-{index:03d}" for index in range(1, 121)]
    selected = [
        {
            "market_id": market_id,
            "market_start_ts": 100 + index,
            "market_end_ts": 200 + index,
        }
        for index, market_id in enumerate(market_ids, start=1)
    ]
    runtime = [
        {
            "market_id": market_id,
            "decision_ts": 150,
            "max_input_ts": 150,
            "side": "UP",
            "action": "BUY_UP_SELL_BEFORE_CLOSE",
        }
        for market_id in market_ids[:40]
    ]
    selected_path = tmp_path / "selected.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(candidate_path, runtime)
    _write_jsonl(baseline_path, runtime)
    manifest = {
        "schema_version": (
            "bigan-v8-adaptive-support-controller-v8-1-future-holdout-"
            "target-free-freeze-manifest-v1"
        ),
        "exact_market_count": 120,
        "decision_freeze_created_ts": 400,
        "target_free_freeze_passed": True,
        "future_target_access_allowed": True,
        "decision_freeze_written_before_target_access": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_model_or_controller_tuning_performed": False,
        "plan": _descriptor(PLAN_PATH),
        "selected_rows": _descriptor(selected_path),
        "candidate_runtime": _descriptor(candidate_path),
        "v6_7_runtime": _descriptor(baseline_path),
        **_v7_0_blocked_safety_fields(),
    }
    path = tmp_path / "freeze.json"
    _write_json(path, manifest)
    return path, market_ids


def _settled_index(
    tmp_path: Path,
    *,
    freeze_path: Path,
    market_ids: list[str],
) -> Path:
    feature_path = tmp_path / "feature.jsonl"
    label_path = tmp_path / "label.jsonl"
    resolution_path = tmp_path / "resolution.jsonl"
    claim_path = tmp_path / "target-claim.json"
    _write_jsonl(feature_path, [{"feature": 1.0}])
    _write_jsonl(label_path, [{"label": 1.0}])
    _write_jsonl(resolution_path, [{"resolution": "UP"}])
    _write_json(claim_path, {"claim": True})
    entries = [
        {
            "market_id": market_id,
            "official_read_only_resolution": True,
            "source_outcome_blind_round_mutated": False,
            "feature_rows": _descriptor(feature_path),
            "label_rows": _descriptor(label_path),
            "resolution_events": _descriptor(resolution_path),
        }
        for market_id in market_ids
    ]
    index = {
        "schema_version": (
            "bigan-v8-adaptive-support-controller-v8-1-future-holdout-"
            "settled-index-v1"
        ),
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "target_access_claim": _descriptor(claim_path),
        "target_access_started_ts": 500,
        "index_finalized_ts": 600,
        "entry_count": 120,
        "entries": entries,
        "official_read_only_resolution": True,
        "source_outcome_blind_rounds_mutated": False,
        "outcomes_used_for_decision_selection_or_tuning": False,
        **_v7_0_blocked_safety_fields(),
    }
    path = tmp_path / "settled.json"
    _write_json(path, index)
    return path


def test_v8_1_future_settlement_is_quarantined_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path, market_ids = _freeze(tmp_path)

    def fake_finalize(rows: list[dict], **_: object) -> list[dict]:
        return [
            {
                "market_id": row["market_id"],
                "settled_corpus_ready": True,
                "index_entry": {
                    "market_id": row["market_id"],
                    "official_read_only_resolution": True,
                    "source_outcome_blind_round_mutated": False,
                },
            }
            for row in rows
        ]

    monkeypatch.setattr(post, "_finalize_selected_rounds", fake_finalize)
    result = post.build_adaptive_support_controller_v8_1_future_settled_index(
        post.AdaptiveSupportControllerV81FutureSettlementConfig(
            run_id="settle",
            output_dir=tmp_path / "out",
            target_free_freeze_manifest_path=freeze_path,
            expected_target_free_freeze_manifest_sha256=_sha256_file(freeze_path),
            implementation_commit="a" * 40,
            target_access_started_ts=1_000,
            settlement_max_wait_seconds=0,
        ),
        provider_factory=lambda: object(),
        clock_ms_fn=lambda: 1_001,
    )
    assert result["report"]["settled_index_ready"] is True
    assert result["report"]["settled_market_count"] == len(market_ids)
    assert result["report"]["source_outcome_blind_rounds_mutated"] is False
    assert result["report"]["capital_at_risk"] is False


def test_v8_1_future_pnl_gate_is_single_use_and_keeps_unlocks_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path, market_ids = _freeze(tmp_path)
    settled_path = _settled_index(
        tmp_path,
        freeze_path=freeze_path,
        market_ids=market_ids,
    )

    def fake_targets(
        decisions: list[dict],
        **_: object,
    ) -> list[dict]:
        return [
            {
                **row,
                "runtime_policy_after_cost_net_pnl_at_frozen_size": 0.01,
                "target_available_only_post_exit_or_official_resolution": True,
                "target_used_as_decision_time_input": False,
            }
            for row in decisions
        ]

    monkeypatch.setattr(post, "_runtime_targets_for_decisions", fake_targets)
    config = post.AdaptiveSupportControllerV81FutureEvaluationConfig(
        run_id="evaluate",
        output_dir=tmp_path / "evaluation",
        target_free_freeze_manifest_path=freeze_path,
        expected_target_free_freeze_manifest_sha256=_sha256_file(freeze_path),
        settled_index_path=settled_path,
        expected_settled_index_sha256=_sha256_file(settled_path),
        runtime_policy_profile_path=RUNTIME_PATH,
        expected_runtime_policy_profile_sha256=_sha256_file(RUNTIME_PATH),
        implementation_commit="b" * 40,
        evaluation_started_ts=700,
    )
    result = post.evaluate_adaptive_support_controller_v8_1_future_pnl_gate(
        config
    )
    report = result["report"]
    assert report["future_pnl_gate_passed"] is True
    assert report["candidate_minus_v6_7_after_cost_pnl"] == 0.0
    assert report["promotion_evidence_eligible"] is False
    assert report["#134_resume_allowed"] is False
    assert report["#146_start_allowed"] is False
    second_config = post.AdaptiveSupportControllerV81FutureEvaluationConfig(
        run_id="evaluate-again",
        output_dir=tmp_path / "evaluation",
        target_free_freeze_manifest_path=freeze_path,
        expected_target_free_freeze_manifest_sha256=_sha256_file(freeze_path),
        settled_index_path=settled_path,
        expected_settled_index_sha256=_sha256_file(settled_path),
        runtime_policy_profile_path=RUNTIME_PATH,
        expected_runtime_policy_profile_sha256=_sha256_file(RUNTIME_PATH),
        implementation_commit="b" * 40,
        evaluation_started_ts=701,
    )
    with pytest.raises(ValueError, match="already been consumed"):
        post.evaluate_adaptive_support_controller_v8_1_future_pnl_gate(
            second_config
        )


def test_v8_1_future_freeze_rejects_insufficient_candidate_support(
    tmp_path: Path,
) -> None:
    freeze_path, _ = _freeze(tmp_path)
    payload = json.loads(freeze_path.read_text())
    candidate_path = Path(payload["candidate_runtime"]["path"])
    rows = [
        json.loads(line)
        for line in candidate_path.read_text().splitlines()
        if line.strip()
    ][:39]
    _write_jsonl(candidate_path, rows)
    payload["candidate_runtime"] = _descriptor(candidate_path)
    _write_json(freeze_path, payload)
    with pytest.raises(ValueError, match="support invalid"):
        post._validated_freeze(
            freeze_path,
            expected_sha256=_sha256_file(freeze_path),
        )
