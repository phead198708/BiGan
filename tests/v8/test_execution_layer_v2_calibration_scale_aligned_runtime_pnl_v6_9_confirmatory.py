from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9_confirmatory import (
    FREEZE_SCHEMA_VERSION,
    _validate_freeze,
    select_v6_9_confirmatory_index_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    build_regime_emergent_target_free_support,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_regime_emergent_pnl_v6_8_evaluation_v1.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _candidate(tmp_path: Path) -> tuple[dict, str]:
    prior_a = tmp_path / "prior-a.jsonl"
    prior_b = tmp_path / "prior-b.jsonl"
    _write_jsonl(prior_a, [{"market_id": "prior-a"}])
    _write_jsonl(prior_b, [{"market_id": "prior-b"}])
    candidate = {
        "candidate_name": "calibration_scale_aligned_runtime_pnl_v6_9",
        "candidate_freeze_created_ts": 1_000,
        "candidate_scoring_frozen": True,
        "strictly_later_outcome_blind_collection_allowed": True,
        "mapping_gate_passed": True,
        "target_free_liveness_gate_passed": True,
        "current_issue229_outcomes_opened": False,
        "future_target_access_allowed": False,
        "profile": {"path": "/profile", "sha256": "a" * 64},
        "mapping_artifact": {"path": "/mapping", "sha256": "b" * 64},
        "liveness_report": {"path": "/liveness", "sha256": "c" * 64},
        "issue229_v6_7_base_selected_rows": _descriptor(prior_a),
        "runtime_target_rows": _descriptor(prior_b),
        **_blocked_safety_fields(),
    }
    path = tmp_path / "candidate.json"
    _write_json(path, candidate)
    return candidate, _sha256_file(path)


def _plan(candidate_sha: str) -> dict:
    return {
        "schema_version": "bigan-v8-v6-9-future-collection-plan-v1",
        "issue_number": 231,
        "candidate_name": "calibration_scale_aligned_runtime_pnl_v6_9",
        "candidate_manifest_sha256": candidate_sha,
        "profile_sha256": "a" * 64,
        "mapping_artifact_sha256": "b" * 64,
        "target_free_liveness_report_sha256": "c" * 64,
        "candidate_freeze_created_ts": 1_000,
        "collection_plan_created_ts": 2_000,
        "minimum_market_start_ts_exclusive": 1_000,
        "target_quality_valid_market_count": 120,
        "maximum_attempted_market_count": 180,
        "batch_round_count": 12,
        "minimum_quality_valid_markets_for_batch_liveness": 6,
        "minimum_guard_accepted_markets_for_batch_liveness": 1,
        "consecutive_zero_action_batch_limit": 1,
        "outcome_blind_collection_only": True,
        "issue229_outcomes_must_remain_sealed": True,
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        "labels_outcomes_or_pnl_opened": False,
        "frozen": True,
        "paper_candidate_allowed": False,
        **_blocked_safety_fields(),
    }


def _index_row(sequence: int, *, valid: bool = True) -> dict:
    start = 2_000 + sequence * 300_000
    return {
        "sequence": sequence,
        "batch_id": f"batch-{(sequence - 1) // 12 + 1}",
        "market_id": f"future-{sequence:03d}",
        "scheduled_round_start_ts": start,
        "market_start_ts": start,
        "market_end_ts": start + 300_000,
        "capture_quality_valid": valid,
        "labels_outcomes_or_pnl_opened": False,
        "raw_resolution_row_count": 0,
        **_blocked_safety_fields(),
    }


def _decision(index: int) -> dict:
    decision_ts = 50_000_000 + index
    return {
        "market_id": f"future-{index + 1:03d}",
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "side": "UP",
        "action": "BUY_UP_SELL_BEFORE_CLOSE",
        "v6_9_calibrated_runtime_expected_pnl_per_contract": 0.05,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
    }


def test_v6_9_confirmatory_selects_earliest_exact_120_without_side_quota(
    tmp_path: Path,
) -> None:
    candidate, candidate_sha = _candidate(tmp_path)
    rows = [_index_row(index, valid=index != 3) for index in range(1, 122)]

    selected, attempted = select_v6_9_confirmatory_index_rows(
        rows,
        candidate_manifest=candidate,
        collection_plan=_plan(candidate_sha),
        candidate_manifest_sha256=candidate_sha,
    )

    assert len(selected) == 120
    assert len(attempted) == 121
    assert 3 not in {row["sequence"] for row in selected}
    assert selected[-1]["sequence"] == 121


def test_v6_9_confirmatory_selection_rejects_prior_market_overlap(
    tmp_path: Path,
) -> None:
    candidate, candidate_sha = _candidate(tmp_path)
    rows = [_index_row(index) for index in range(1, 121)]
    rows[0]["market_id"] = "prior-a"

    with pytest.raises(ValueError, match="missing or overlapping"):
        select_v6_9_confirmatory_index_rows(
            rows,
            candidate_manifest=candidate,
            collection_plan=_plan(candidate_sha),
            candidate_manifest_sha256=candidate_sha,
        )


def test_v6_9_freeze_accepts_regime_emergent_one_sided_support(
    tmp_path: Path,
) -> None:
    candidate, candidate_sha = _candidate(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    plan = _plan(candidate_sha)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    selected = [_index_row(index) for index in range(1, 121)]
    decisions = [_decision(index) for index in range(40)]
    selected_path = tmp_path / "selected.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    profile = json.loads(PROFILE_PATH.read_text())
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=120,
        expected_window_market_count=120,
        required_total_market_count=40,
        score_field="v6_9_calibrated_runtime_expected_pnl_per_contract",
    )
    decision = {
        "selected_window_market_ids": [row["market_id"] for row in selected],
        "target_free_support": support,
        "future_target_access_allowed": True,
    }
    decision_path = tmp_path / "decision.json"
    _write_json(decision_path, decision)
    freeze = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "role": "future_confirmatory",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "candidate_manifest": _descriptor(candidate_path),
        "collection_plan": _descriptor(plan_path),
        "selected_window_rows": _descriptor(selected_path),
        "v6_9_selected_decisions": _descriptor(decisions_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "future_target_access_allowed": True,
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "paper_candidate_allowed": False,
        **_blocked_safety_fields(),
    }

    _validate_freeze(freeze, profile=profile, profile_path=PROFILE_PATH)


def test_v6_9_freeze_remains_fail_closed_below_total_support(
    tmp_path: Path,
) -> None:
    candidate, candidate_sha = _candidate(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, _plan(candidate_sha))
    selected = [_index_row(index) for index in range(1, 121)]
    decisions = [_decision(index) for index in range(39)]
    selected_path = tmp_path / "selected.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    _write_jsonl(selected_path, selected)
    _write_jsonl(decisions_path, decisions)
    profile = json.loads(PROFILE_PATH.read_text())
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=120,
        expected_window_market_count=120,
        required_total_market_count=40,
        score_field="v6_9_calibrated_runtime_expected_pnl_per_contract",
    )
    decision_path = tmp_path / "decision.json"
    _write_json(
        decision_path,
        {
            "selected_window_market_ids": [row["market_id"] for row in selected],
            "target_free_support": support,
            "future_target_access_allowed": False,
        },
    )
    freeze = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "role": "future_confirmatory",
        "evaluation_profile": _descriptor(PROFILE_PATH),
        "candidate_manifest": _descriptor(candidate_path),
        "collection_plan": _descriptor(plan_path),
        "selected_window_rows": _descriptor(selected_path),
        "v6_9_selected_decisions": _descriptor(decisions_path),
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "future_target_access_allowed": True,
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "paper_candidate_allowed": False,
        **_blocked_safety_fields(),
    }

    with pytest.raises(ValueError, match="decision-freeze evidence mismatch"):
        _validate_freeze(freeze, profile=profile, profile_path=PROFILE_PATH)
