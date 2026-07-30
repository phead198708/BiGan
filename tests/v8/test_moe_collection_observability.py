from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_collection_observability import (
    CANDIDATE_BUNDLE_HASH,
    MAXIMUM_ATTEMPTS,
    REPORT_SAFETY,
    REQUIRED_CURRENT_RAW_FILES,
    TARGET_QUALITY_VALID_MARKETS,
    _allowed_current_path,
    _assert_outcome_free_decision_rows,
    _attribution_report,
    _distribution_report,
    _health_report,
    _load_development_distribution_reference,
    _load_runtime_bundle,
    build_evaluation_dry_run_report,
    build_finalization_checklist,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _attempt(
    *,
    index: int,
    quality_valid: bool,
    provider_failed: bool = False,
    retry_used: bool = False,
) -> dict[str, object]:
    decision = {
        "market_id": f"market-{index}",
        "decision_ts": 1_700_000_000_000 + index,
        "requested_route": "low_vol" if index == 2 else "bullish",
        "actual_model_used": (
            "global_baseline_fallback" if index == 2 else "moe_expert_bullish"
        ),
        "expert_id": "moe_expert_low_vol" if index == 2 else "moe_expert_bullish",
        "expert_training_support": 18 if index == 2 else 40,
        "expert_available": index != 2,
        "fallback_used": index == 2,
        "selected_side": "UP" if index == 1 else None,
        "accepted": index == 1,
        "rejected": index != 1,
        "provider_health_score": None,
        "missing_feature_count": 3,
        "router_inputs": {},
        "regime_bucket": {
            "btc_return_regime": "bullish",
            "volatility_bucket": "low" if index == 2 else "medium",
        },
        "decision_source": "post_capture_frozen_artifact_replay",
    }
    return {
        "attempt_index": index,
        "attempt_id": f"attempt-{index}",
        "market_id": f"market-{index}",
        "market_start_ts": 1_700_000_000_000 + index,
        "capture_manifest_sha256": str(index) * 64,
        "quality": {
            "quality_valid": quality_valid,
            "invalid_reason_codes": (
                [] if quality_valid else ["provider_capture_complete_failed"]
            ),
        },
        "data_quality": {
            "raw_market_capture_coverage": True,
            "orderbook_coverage": True,
            "paired_executable_ask_coverage": 1.0,
            "trade_tape_available": not provider_failed,
            "btc_feature_coverage": 1.0,
            "chainlink_reference_coverage": True,
            "causality_violation_count": 0,
            "missing_feature_count": 3,
            "missing_feature_counts": {"recent_trade_volume": 3},
        },
        "provider_failed": provider_failed,
        "retry_used": retry_used,
        "decision_rows": [decision],
        "market_observation": {
            **decision,
            "market_start_ts": 1_700_000_000_000 + index,
            "combined_spread_bps": 100.0 + index,
            "total_liquidity_depth": 20.0 + index,
            "time_to_close_seconds": 600.0,
            "btc_volatility_15m": 0.0003,
        },
    }


def test_current_reader_allowlist_excludes_target_artifacts() -> None:
    assert all(
        token not in filename
        for filename in REQUIRED_CURRENT_RAW_FILES
        for token in ("resolution", "settlement", "pnl", "label")
    )
    for filename in (
        "raw_polymarket_resolutions.jsonl",
        "settlement.json",
        "realized_pnl.json",
        "label_rows.jsonl",
    ):
        with pytest.raises(ValueError, match="forbidden current confirmatory"):
            _allowed_current_path(Path(filename))


def test_health_report_is_outcome_free_missing_safe_and_observational() -> None:
    report = _health_report(
        [
            _attempt(index=1, quality_valid=True),
            _attempt(
                index=2,
                quality_valid=False,
                provider_failed=True,
                retry_used=True,
            ),
        ],
        created_at="2026-07-30T00:00:00+00:00",
        hash_chain_status="valid",
    )

    assert report["progress"]["attempts_consumed"] == 2
    assert report["progress"]["maximum_attempts"] == MAXIMUM_ATTEMPTS
    assert report["progress"]["quality_valid_market_count"] == 1
    assert report["progress"]["target_quality_valid_market_count"] == (
        TARGET_QUALITY_VALID_MARKETS
    )
    assert report["progress"]["remaining_quality_valid_markets"] == 799
    assert report["attempt_health"]["invalid_reason_distribution"] == {
        "provider_capture_complete_failed": 1
    }
    assert report["attempt_health"]["provider_failure_rate"] == 0.5
    assert report["attempt_health"]["retry_rate"] == 0.5
    assert report["data_quality"]["missing_feature_count_total"] == 6
    assert report["data_quality"]["missing_feature_counts"] == {
        "recent_trade_volume": 6
    }
    assert report["data_quality"]["missing_values_encoded_as_zeros"] is False
    assert report["monitoring_influences_collection_decisions"] is False
    assert report["outcomes_accessed"] is False
    assert report["settlement_accessed"] is False
    assert report["pnl_accessed"] is False
    assert report["safety"] == REPORT_SAFETY
    assert not any(report["safety"].values())


def test_attribution_is_target_free_and_reconciles_expert_fallback() -> None:
    attempts = [
        _attempt(index=1, quality_valid=True),
        _attempt(index=2, quality_valid=True),
    ]
    report = _attribution_report(
        attempts,
        created_at="2026-07-30T00:00:00+00:00",
    )

    assert report["decision_row_count"] == 2
    assert report["native_expert_decision_count"] == 1
    assert report["fallback_decision_count"] == 1
    assert report["fallback_share"] == 0.5
    assert report["current_confirmatory_outcomes_accessed"] is False
    assert report["current_confirmatory_settlement_accessed"] is False
    assert report["current_confirmatory_pnl_accessed"] is False
    _assert_outcome_free_decision_rows(report["decision_rows"])

    bad_row = dict(report["decision_rows"][0])
    bad_row["unit_pnl"] = 1.0
    with pytest.raises(ValueError, match="forbidden target fields"):
        _assert_outcome_free_decision_rows([bad_row])


def test_distribution_report_is_diagnostic_only() -> None:
    development = [
        _attempt(index=1, quality_valid=True)["market_observation"],
        _attempt(index=2, quality_valid=True)["market_observation"],
    ]
    confirmatory = [_attempt(index=1, quality_valid=True)["market_observation"]]
    report = _distribution_report(
        development=development,
        confirmatory=confirmatory,
        created_at="2026-07-30T00:00:00+00:00",
    )

    assert report["development_market_count"] == 2
    assert report["confirmatory_market_count"] == 1
    assert report["diagnostic_only"] is True
    assert report["route_filtering_allowed"] is False
    assert report["collection_stop_or_change_allowed"] is False
    assert report["monitoring_influences_collection_decisions"] is False
    assert report["outcomes_accessed"] is False
    assert report["settlement_accessed"] is False
    assert report["pnl_accessed"] is False


def test_frozen_runtime_bundle_loads_without_changing_bytes() -> None:
    bundle = _load_runtime_bundle(REPO_ROOT)

    assert bundle["graph"]["bundle_hash"] == CANDIDATE_BUNDLE_HASH
    assert set(bundle["route_support"]) == {
        "bearish",
        "bullish",
        "high_vol",
        "low_vol",
    }
    assert bundle["route_available"] == {
        "bearish": True,
        "bullish": True,
        "high_vol": True,
        "low_vol": False,
    }
    assert set(bundle["experts"]) == {"bearish", "bullish", "high_vol"}


def test_development_distribution_reference_is_portable_and_target_free() -> None:
    rows = _load_development_distribution_reference(repository_root=REPO_ROOT)

    assert len(rows) == 113
    _assert_outcome_free_decision_rows(rows)
    assert {str(row["requested_route"]) for row in rows} == {
        "bearish",
        "bullish",
        "high_vol",
        "low_vol",
    }
    assert all(not Path(str(row["market_id"])).is_absolute() for row in rows)


def test_evaluation_dry_run_is_deterministic_and_emits_no_gate_result(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    kwargs = {
        "repository_root": REPO_ROOT,
        "created_at": "2026-07-30T00:00:00+00:00",
    }
    first = build_evaluation_dry_run_report(output_path=first_path, **kwargs)
    second = build_evaluation_dry_run_report(output_path=second_path, **kwargs)

    assert first == second
    assert first["dry_run_passed"] is True
    assert all(first["checks"].values())
    assert first["confirmatory_gate_result"] is None
    assert first["promotion_or_pass_result_emitted"] is False
    assert first["current_confirmatory_artifacts_read"] is False
    assert first["current_confirmatory_outcomes_accessed"] is False
    assert first["development_only_forever"] is True
    assert first["promotion_evidence_eligible"] is False
    assert sha256_file(first_path) == sha256_file(second_path)
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert first_path.with_suffix(".sha256").read_text().startswith(
        sha256_file(first_path)
    )


def test_finalization_checklist_keeps_outcome_access_blocked(
    tmp_path: Path,
) -> None:
    output = tmp_path / "checklist.json"
    report = build_finalization_checklist(
        output_path=output,
        created_at="2026-07-30T00:00:00+00:00",
    )

    before = report["before_outcome_access"]
    after = report["after_outcome_access"]
    assert before["exact_800_population_frozen"] is False
    assert before["candidate_rows_equal_800"] is False
    assert before["baseline_rows_equal_800"] is False
    assert before["paired_rows_equal_800"] is False
    assert before["outcome_access_allowed_now"] is False
    assert before["partial_incremental_or_selective_open_forbidden"] is True
    assert after["official_settlement_only"] is True
    assert after["gate_evaluation_exactly_once"] is True
    assert after["population_changes_allowed"] is False
    assert after["not_executed"] is True
    assert report["fresh_collection_started"] is True
    assert report["fresh_outcomes_opened"] is False
    assert not any(report["safety"].values())
