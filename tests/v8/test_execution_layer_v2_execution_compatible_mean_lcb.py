from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_execution_compatible_mean_lcb import (
    ExecutionCompatibleMeanLCBPrecollectionFreezeConfig,
    _capture_quality_audit,
    _execution_compatibility_audit,
    _role_for_rank,
    freeze_execution_compatible_mean_lcb_precollection,
    validate_execution_compatible_mean_lcb_feature_contract,
    validate_execution_compatible_mean_lcb_protocol,
)
from bigan.v8.polymarket.training.execution_layer_v2_execution_compatible_mean_lcb_fit import (
    _apply_expected_mean_lcb_scores,
    _expected_mean_lcb_artifact,
    _validate_execution_compatibility_report,
    _validate_role_rows,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_execution_compatible_mean_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_execution_compatible_mean_lcb_feature_contract_v1.json"
)


def test_issue173_protocol_is_execution_compatible_and_frozen() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    validate_execution_compatible_mean_lcb_protocol(protocol)
    roles = protocol["role_assignment"]
    collector = protocol["collector_contract"]
    calibration = protocol["expected_mean_lcb_protocol"]

    assert roles["development_train_market_count"] == 60
    assert roles["development_calibration_market_count"] == 30
    assert roles["confirmatory_validation_market_count"] == 30
    assert collector["orderbook_snapshot_interval_seconds"] == 1.0
    assert collector["maximum_selected_side_book_staleness_ms"] == 2_000.0
    assert collector["maximum_opposite_side_book_staleness_ms"] == 2_000.0
    assert calibration["estimand"] == "conditional_expected_cost_aware_net_return"
    assert calibration["bootstrap_unit"] == "market_id"
    assert calibration["individual_outcome_quantile_subtraction_enabled"] is False
    assert protocol["safety"]["v8_execution_handoff_allowed"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["collector_contract"]["orderbook_snapshot_interval_seconds"] = 30.0
    with pytest.raises(ValueError, match="execution_compatible_collection"):
        validate_execution_compatible_mean_lcb_protocol(drifted)


def test_issue173_feature_contract_is_hash_pinned_and_causal() -> None:
    contract = _load_json(FEATURE_CONTRACT_PATH)
    validate_execution_compatible_mean_lcb_feature_contract(
        contract,
        expected_parent_protocol_sha256=_sha256(PROTOCOL_PATH),
    )
    assert contract["execution_compatibility_must_pass_before_label_access"] is True
    assert contract["individual_outcome_quantile_subtraction_enabled"] is False
    assert contract["uses_confirmatory_validation_labels_for_tuning"] is False
    assert contract["market_implied_probability_used_as_direct_fair_value_ev"] is False


def test_precollection_freeze_pins_60_30_30_roles_and_prior_exclusions(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior.jsonl"
    _write_jsonl(
        prior,
        [
            {"market_id": "prior-a", "decision_ts": 1_000_000},
            {"market_id": "prior-b", "decision_ts": 1_300_000},
        ],
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"blocked"}\n', encoding="utf-8")
    result = freeze_execution_compatible_mean_lcb_precollection(
        ExecutionCompatibleMeanLCBPrecollectionFreezeConfig(
            run_id="issue173-freeze",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            git_commit="a" * 40,
            prior_market_registry_pins=((prior, _sha256(prior)),),
            prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
            expected_prior_unique_market_count=2,
        )
    )
    manifest = result["manifest"]
    assert manifest["role_plan"] == [
        {
            "role": "development_train",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": 60,
        },
        {
            "role": "development_calibration",
            "valid_market_rank_start": 61,
            "valid_market_rank_end": 90,
        },
        {
            "role": "confirmatory_validation",
            "valid_market_rank_start": 91,
            "valid_market_rank_end": 120,
        },
    ]
    assert manifest["collector_contract"]["orderbook_snapshot_interval_seconds"] == 1.0
    assert manifest["expected_mean_lcb_protocol"]["bootstrap_unit"] == "market_id"
    assert manifest["collection_started"] is False
    assert manifest["source_model_candidate_eligible"] is False

    assert [_role_for_rank(rank) for rank in (1, 60, 61, 90, 91, 120)] == [
        "development_train",
        "development_train",
        "development_calibration",
        "development_calibration",
        "confirmatory_validation",
        "confirmatory_validation",
    ]


def test_execution_compatibility_fails_before_label_access_on_stale_book(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
    _write_jsonl(feature_path, [_feature_row(book_staleness_ms=1_999.0)])
    collector = _load_json(PROTOCOL_PATH)["collector_contract"]

    passed = _execution_compatibility_audit(
        corpus_dir=corpus_dir,
        collector_contract=collector,
    )
    assert passed["blocking_reason_codes"] == []
    assert passed["execution_compatible_row_count"] == 1
    assert passed["labels_or_outcomes_opened"] is False

    _write_jsonl(feature_path, [_feature_row(book_staleness_ms=2_001.0)])
    blocked = _execution_compatibility_audit(
        corpus_dir=corpus_dir,
        collector_contract=collector,
    )
    assert blocked["execution_compatible_row_count"] == 0
    assert blocked["blocking_reason_codes"] == [
        "execution_compatibility_down_book_staleness_exceeded",
        "execution_compatibility_up_book_staleness_exceeded",
    ]
    assert blocked["labels_or_outcomes_opened"] is False


def test_capture_audit_proves_runtime_collector_interval_matches_freeze() -> None:
    collector = _load_json(PROTOCOL_PATH)["collector_contract"]
    capture = {
        "capture_start_boundary_validation_passed": True,
        "scheduled_round_start_ts": 2_000_000,
        "raw_polymarket_market_count": 1,
        "provider_raw_orderbook_snapshot_count": 100,
        "training_sampled_orderbook_row_count": 100,
        "raw_btc_candle_row_count": 5,
        "raw_chainlink_price_row_count": 100,
        "orderbook_snapshot_interval_seconds": 1.0,
        "public_provider_timeout_seconds": 330.0,
        "public_provider_http_timeout_seconds": 5.0,
        "reject_reason_counts": {},
    }
    assert _capture_quality_audit(
        capture, collector_contract=collector
    )["reason_codes"] == []

    capture["orderbook_snapshot_interval_seconds"] = 30.0
    assert "collector_orderbook_snapshot_interval_contract_failed" in (
        _capture_quality_audit(capture, collector_contract=collector)["reason_codes"]
    )


def test_expected_mean_lcb_is_market_grouped_deterministic_and_not_outcome_quantile() -> None:
    protocol = _load_json(PROTOCOL_PATH)
    train_oof = _oof_rows()
    calibration = _calibration_rows()

    first = _expected_mean_lcb_artifact(
        calibration,
        train_oof_predictions=train_oof,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    second = _expected_mean_lcb_artifact(
        calibration,
        train_oof_predictions=train_oof,
        protocol=protocol,
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )

    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    assert first["estimand"] == "conditional_expected_cost_aware_net_return"
    assert first["method"] == (
        "market_grouped_bootstrap_mean_residual_upper_confidence_bound"
    )
    assert first["individual_outcome_quantile_subtraction_enabled"] is False
    assert "one_sided_quantile" not in first
    assert len(first["calibration_groups"]) == 6
    assert all(
        group["market_grouped_bootstrap"]["bootstrap_resample_count"] == 2_000
        for group in first["calibration_groups"].values()
    )
    assert first["source_model_candidate_eligible"] is False
    assert first["v8_execution_handoff_allowed"] is False

    prediction = {
        "market_id": "future-market",
        "decision_ts": 2_000_000_000_000,
        "action": "BUY_UP_HOLD_TO_SETTLEMENT",
        "action_family": "HOLD_TO_SETTLEMENT",
        "raw_family_expected_net_return": 0.08,
    }
    scored = _apply_expected_mean_lcb_scores([prediction], lcb_artifact=first)[0]
    assert scored["ranking_score_source"] == (
        "calibration_only_expected_mean_residual_confidence_bound"
    )
    assert scored["expected_mean_lcb_score_bucket"] in {"low", "middle", "high"}
    assert scored["expected_mean_lcb_net_return"] == pytest.approx(
        scored["raw_family_expected_net_return"]
        - scored["expected_mean_residual_upper_confidence_bound"]
    )


def test_expected_mean_lcb_uses_predeclared_family_fallback_for_low_group_support() -> None:
    artifact = _expected_mean_lcb_artifact(
        _calibration_rows(score_mode="single_bucket"),
        train_oof_predictions=_oof_rows(),
        protocol=_load_json(PROTOCOL_PATH),
        feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
    )
    low_support = [
        group
        for group in artifact["calibration_groups"].values()
        if not group["group_support_passed"]
    ]
    assert low_support
    assert all(
        group["penalty_source"] == "family_level_mean_residual_ci_fallback"
        and group["shrinkage_group_weight"] == 0.0
        for group in low_support
    )


def test_role_rows_require_exact_outcome_blind_execution_compatible_60_30_30() -> None:
    rows = []
    for index in range(120):
        rows.append(
            {
                "market_id": f"market-{index:03d}",
                "selection_rank": index + 1,
                "role": (
                    "development_train"
                    if index < 60
                    else "development_calibration"
                    if index < 90
                    else "confirmatory_validation"
                ),
                "labels_or_outcomes_opened_for_role_assignment": False,
                "execution_compatibility_validated_before_label_access": True,
            }
        )
    _validate_role_rows(rows)

    rows[-1]["execution_compatibility_validated_before_label_access"] = False
    with pytest.raises(ValueError, match="execution compatibility"):
        _validate_role_rows(rows)


def test_zero_execution_compatibility_failures_is_accepted_not_treated_as_missing() -> None:
    _validate_execution_compatibility_report(
        {
            "execution_compatibility_validated_before_label_access": True,
            "selected_market_failure_count": 0,
            "selected_market_count": 120,
        }
    )
    with pytest.raises(ValueError, match="before label access"):
        _validate_execution_compatibility_report(
            {
                "execution_compatibility_validated_before_label_access": True,
                "selected_market_failure_count": 1,
                "selected_market_count": 120,
            }
        )


def _feature_row(*, book_staleness_ms: float) -> dict:
    features = {
        "time_to_close_seconds": 180.0,
        "up_bid": 0.49,
        "up_ask": 0.51,
        "down_bid": 0.49,
        "down_ask": 0.51,
    }
    for side in ("up", "down"):
        features.update(
            {
                f"{side}_spread_bps": 400.0,
                f"{side}_queue_fill_probability_proxy": 0.8,
                f"{side}_book_staleness_ms": book_staleness_ms,
                f"{side}_liquidity_depth": 25.0,
                f"{side}_executable_ask_notional": 5.0,
            }
        )
    return {
        "market_id": "market-001",
        "decision_ts": 2_000_000,
        "max_input_ts": 1_999_999,
        "features": features,
    }


def _oof_rows() -> list[dict]:
    rows = []
    for family_index, family in enumerate(
        ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")
    ):
        for index in range(60):
            rows.append(
                {
                    "market_id": f"train-{index:03d}",
                    "action_family": family,
                    "oof_raw_prediction": -0.06 + index * 0.002 + family_index * 0.001,
                }
            )
    return rows


def _calibration_rows(*, score_mode: str = "spread") -> list[dict]:
    rows = []
    for family_index, family in enumerate(
        ("HOLD_TO_SETTLEMENT", "SELL_BEFORE_CLOSE")
    ):
        for index in range(30):
            raw = 0.02 if score_mode == "single_bucket" else -0.05 + index * 0.004
            residual = 0.005 + (index % 3) * 0.001 + family_index * 0.0005
            rows.append(
                {
                    "market_id": f"calibration-{index:03d}",
                    "action_family": family,
                    "raw_family_expected_net_return": raw,
                    "target_net_pnl_per_contract": raw - residual,
                }
            )
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
