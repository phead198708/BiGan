from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    PUpSemanticCompatibilityV67Config,
    build_v6_7_target_free_candidate_rows,
    run_p_up_semantic_compatibility_v6_7,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json"
)
SOURCE_ROOT = (
    PROJECT_ROOT
    / "examples/v8/polymarket_runs/"
    "policy-selected-runtime-pnl-v6-6-fresh-calibration-freeze-20260720T232530Z"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text())


def _action_row(
    *,
    market_id: str,
    decision_ts: int,
    action: str,
    p_up_disagreement: bool,
    staleness_ms: float = 100.0,
) -> dict[str, object]:
    side = "UP" if "BUY_UP" in action else "DOWN"
    selected_probability = 0.40 if p_up_disagreement else 0.60
    if side == "DOWN":
        selected_probability = 1.0 - selected_probability
    return {
        "market_id": market_id,
        "market_slug": f"slug-{market_id}",
        "decision_ts": decision_ts,
        "market_close_ts": decision_ts + 240_000,
        "max_input_ts": decision_ts - 1,
        "action": action,
        "side": side,
        "p_up": 0.40,
        "p_down": 0.60,
        "selected_side_probability": selected_probability,
        "decision_time_features": {
            "execution_price": 0.39,
            "selected_side_executable_ask_notional": 1.0,
            "selected_side_executable_bid_notional": 1.0,
            "selected_side_liquidity_depth": 2.0,
        },
        "microstructure_snapshot": {
            "entry_ask": 0.39,
            "entry_bid": 0.38,
            "spread_bps": 100.0,
            "book_staleness_ms": staleness_ms,
            "queue_fill_proxy": 0.9,
            "time_to_close_seconds": 240.0,
        },
        "reference_price_feature_provenance": {"provenance_valid": True},
    }


def _prediction(row: dict[str, object], *, score: float) -> dict[str, object]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row["decision_ts"],
        "action": row["action"],
        "mean_ev_lower_confidence_bound": score,
        "calibrated_action_expected_net_return": score + 0.01,
        "raw_pairwise_rank_score": score + 0.02,
        "p_up_action_disagreement": row["selected_side_probability"] < 0.5,
    }


def test_v6_7_profile_freezes_p_up_as_diagnostic_only() -> None:
    profile = _profile()
    validate_p_up_semantic_compatibility_v6_7_profile(profile)

    invalid = copy.deepcopy(profile)
    invalid["p_up_semantics"][
        "market_implied_probability_used_as_direct_fair_value_ev"
    ] = True
    with pytest.raises(ValueError, match="semantics"):
        validate_p_up_semantic_compatibility_v6_7_profile(invalid)


def test_v6_7_accepts_p_up_disagreement_but_keeps_microstructure_guard() -> None:
    disagreed = _action_row(
        market_id="market-up",
        decision_ts=1_000,
        action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_disagreement=True,
    )
    stale = _action_row(
        market_id="market-stale",
        decision_ts=2_000,
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        p_up_disagreement=False,
        staleness_ms=2_001.0,
    )
    candidates, summary = build_v6_7_target_free_candidate_rows(
        [_prediction(disagreed, score=0.05), _prediction(stale, score=0.10)],
        action_rows=[disagreed, stale],
        profile=_profile(),
    )

    assert len(candidates) == 1
    assert candidates[0]["p_up_action_disagreement"] is True
    assert candidates[0]["p_up_action_disagreement_diagnostic_only"] is True
    assert candidates[0]["p_up_side_alignment_filter_enabled"] is False
    assert summary["excluded_reason_distribution"]["execution_book_stale"] == 1


def test_v6_7_selection_is_one_row_per_market_and_target_free() -> None:
    first = _action_row(
        market_id="market-1",
        decision_ts=1_000,
        action="BUY_UP_SELL_BEFORE_CLOSE",
        p_up_disagreement=True,
    )
    second = _action_row(
        market_id="market-1",
        decision_ts=2_000,
        action="BUY_DOWN_SELL_BEFORE_CLOSE",
        p_up_disagreement=False,
    )
    candidates, _ = build_v6_7_target_free_candidate_rows(
        [_prediction(first, score=0.05), _prediction(second, score=0.08)],
        action_rows=[first, second],
        profile=_profile(),
    )
    selected = select_v6_7_target_free_rows(candidates, profile=_profile())

    assert len(selected) == 1
    assert selected[0]["action"] == "BUY_DOWN_SELL_BEFORE_CLOSE"
    assert selected[0]["target_fields_used_for_selection"] is False
    assert selected[0]["source_score_mutated"] is False


def test_v6_7_real_exact_60_canary_passes_without_unlock(tmp_path: Path) -> None:
    source_manifest = SOURCE_ROOT / "v6_6_fresh_calibration_prediction_freeze_manifest.json"
    predictions = SOURCE_ROOT / "v6_2_target_free_predictions.jsonl"
    action_rows = SOURCE_ROOT / "v6_6_target_free_five_action_rows.jsonl"
    legacy_replay = SOURCE_ROOT / "v6_2_outcome_blind_guard_replay.jsonl"
    result = run_p_up_semantic_compatibility_v6_7(
        PUpSemanticCompatibilityV67Config(
            run_id="test-v6-7-exact-60",
            output_dir=tmp_path,
            profile_path=PROFILE_PATH,
            expected_profile_sha256=_sha256(PROFILE_PATH),
            source_freeze_manifest_path=source_manifest,
            expected_source_freeze_manifest_sha256=_sha256(source_manifest),
            predictions_path=predictions,
            expected_predictions_sha256=_sha256(predictions),
            five_action_rows_path=action_rows,
            expected_five_action_rows_sha256=_sha256(action_rows),
            legacy_guard_replay_path=legacy_replay,
            expected_legacy_guard_replay_sha256=_sha256(legacy_replay),
            implementation_commit="a" * 40,
            candidate_freeze_created_ts=2_000_000_000_000,
        )
    )
    report = json.loads(Path(result["report_path"]).read_text())
    manifest = json.loads(Path(result["manifest_path"]).read_text())

    assert result["target_free_support_gate_passed"] is True
    assert result["selected_side_count"] == {"DOWN": 40, "UP": 20}
    assert report["v6_7_selected_p_up_disagreement_count"] == 48
    assert report["labels_outcomes_resolution_or_pnl_opened"] is False
    assert report["hard_execution_safety_thresholds_unchanged"] is True
    assert manifest["candidate_scoring_frozen"] is True
    assert manifest["strictly_later_outcome_blind_collection_allowed"] is True
    for key in (
        "v8_execution_handoff_allowed",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "#134_resume_allowed",
        "#146_start_allowed",
    ):
        assert report[key] is False
        assert manifest[key] is False
