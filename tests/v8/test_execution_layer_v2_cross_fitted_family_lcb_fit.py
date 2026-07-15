from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb_fit import (
    CrossFittedFamilyLCBFitConfig,
    fit_cross_fitted_family_lcb,
    validate_cross_fitted_family_lcb_feature_contract,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_feature_contract_v1.json"
)
ACTIONS = (
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "NO_TRADE",
)


def test_feature_contract_is_frozen_causal_and_not_direct_market_ev() -> None:
    contract = _load_json(FEATURE_CONTRACT_PATH)
    validate_cross_fitted_family_lcb_feature_contract(
        contract,
        expected_parent_protocol_sha256=_sha256(PROTOCOL_PATH),
    )
    assert contract["market_implied_probability_used_as_conditioning_feature"] is True
    assert contract["market_implied_probability_used_as_direct_fair_value_ev"] is False
    assert contract["uses_confirmatory_validation_labels_for_tuning"] is False
    assert contract["target_includes_fees_slippage_and_liquidity_impact"] is True

    invalid = json.loads(json.dumps(contract))
    invalid["market_implied_probability_used_as_direct_fair_value_ev"] = True
    with pytest.raises(ValueError, match="market_probability_semantics"):
        validate_cross_fitted_family_lcb_feature_contract(
            invalid,
            expected_parent_protocol_sha256=_sha256(PROTOCOL_PATH),
        )


def test_cross_fitted_family_lcb_fit_uses_frozen_roles_and_stays_fail_closed(
    tmp_path: Path,
) -> None:
    role_manifest = _fit_fixture(tmp_path)
    result = fit_cross_fitted_family_lcb(
        CrossFittedFamilyLCBFitConfig(
            run_id="issue172-fit",
            output_dir=tmp_path / "fit-runs",
            role_assignment_manifest_path=role_manifest,
            expected_role_assignment_manifest_sha256=_sha256(role_manifest),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
        )
    )

    split = _load_json(result["split_manifest_path"])
    assert split["roles"]["development_train"]["market_count"] == 40
    assert split["roles"]["development_calibration"]["market_count"] == 20
    assert split["roles"]["confirmatory_validation"]["market_count"] == 30
    assert split["role_market_overlap_count"] == 0
    assert split["chronology_validation_passed"] is True
    assert split["feature_causality_violation_count"] == 0

    training = _load_json(result["training_report_path"])
    assert result["training_report_path"].name == "cross_fit_training_report.json"
    assert training["cross_fit"]["fold_count"] == 5
    assert training["cross_fit"]["market_count"] == 40
    assert training["cross_fit"]["oof_prediction_count"] == 160
    assert training["cross_fit"]["oof_prediction_coverage_complete"] is True
    assert all(
        fold["training_market_count"] == 32
        and fold["validation_market_count"] == 8
        and fold["market_overlap_count"] == 0
        for fold in training["cross_fit"]["fold_reports"]
    )
    assert training["confirmatory_labels_opened_before_model_and_lcb_freeze"] is False
    assert training["uses_confirmatory_validation_labels_for_tuning"] is False
    development_freeze = _load_json(
        Path(training["development_fit_freeze_manifest"]["path"])
    )
    assert development_freeze["confirmatory_labels_opened_before_this_freeze"] is False
    assert development_freeze["uses_confirmatory_validation_labels_for_tuning"] is False
    calibration_report = _load_json(result["calibration_report_path"])
    assert result["calibration_report_path"].name == (
        "conformal_lcb_calibration_report.json"
    )
    assert calibration_report["source_split"] == "development_calibration_only"
    assert calibration_report[
        "confirmatory_labels_opened_before_calibration_freeze"
    ] is False
    leakage_audit = _load_json(result["leakage_audit_path"])
    assert result["leakage_audit_path"].name == "leakage_and_role_audit.json"
    assert leakage_audit["leakage_and_role_audit_passed"] is True
    assert leakage_audit["prior_market_overlap_count"] == 0
    assert leakage_audit["forbidden_inference_field_violation_count"] == 0

    freeze = result["freeze_manifest"]
    lcb = _load_json(Path(freeze["family_lcb_calibration_artifact"]["path"]))
    assert lcb["source_split"] == "development_calibration_only"
    assert lcb["families"]["HOLD_TO_SETTLEMENT"]["calibration_row_count"] == 40
    assert lcb["families"]["SELL_BEFORE_CLOSE"]["calibration_row_count"] == 40
    assert lcb["uses_confirmatory_validation_labels_for_tuning"] is False

    validation = result["validation_report"]
    assert result["validation_report_path"].name == "confirmatory_validation_report.json"
    assert validation["confirmatory_labels_used_for_tuning"] is False
    assert validation["candidate_metrics"]["accepted_unique_market_count"] <= 30
    assert freeze["future_unseen_evaluation_required"] is True
    assert freeze["source_model_candidate_eligible"] is False
    assert freeze["freeze_ready"] is False
    assert freeze["promotion_evidence_eligible"] is False
    assert freeze["v8_execution_handoff_allowed"] is False
    assert freeze["#134_resume_allowed"] is False
    assert freeze["#146_start_allowed"] is False
    assert freeze["paper_only"] is True
    assert freeze["capital_at_risk"] is False
    assert result["freeze_manifest_path"].name == "candidate_freeze_manifest.json"


def test_fit_rejects_feature_contract_not_frozen_before_collection(
    tmp_path: Path,
) -> None:
    role_manifest = _fit_fixture(tmp_path)
    copied_contract = tmp_path / "copied_feature_contract.json"
    copied_contract.write_bytes(FEATURE_CONTRACT_PATH.read_bytes())

    with pytest.raises(
        ValueError,
        match="feature contract path does not match precollection freeze",
    ):
        fit_cross_fitted_family_lcb(
            CrossFittedFamilyLCBFitConfig(
                run_id="issue172-feature-contract-swap",
                output_dir=tmp_path / "fit-runs",
                role_assignment_manifest_path=role_manifest,
                expected_role_assignment_manifest_sha256=_sha256(role_manifest),
                feature_contract_path=copied_contract,
                expected_feature_contract_sha256=_sha256(copied_contract),
            )
        )


def _fit_fixture(tmp_path: Path) -> Path:
    role_rows = []
    for index in range(90):
        role = (
            "development_train"
            if index < 40
            else "development_calibration"
            if index < 60
            else "confirmatory_validation"
        )
        market_id = f"market-{index + 1:03d}"
        decision_ts = 1_800_000_000_000 + index * 300_000
        outcome = "UP" if index % 2 == 0 else "DOWN"
        corpus_dir = tmp_path / "training" / "polymarket" / market_id
        corpus_dir.mkdir(parents=True)
        feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
        label_path = corpus_dir / "polymarket_label_rows.jsonl"
        metadata_path = corpus_dir / "polymarket_market_metadata.jsonl"
        resolution_path = corpus_dir / "polymarket_resolution_events.jsonl"
        feature_row = _feature_row(market_id, decision_ts, outcome)
        _write_jsonl(feature_path, [feature_row])
        _write_jsonl(
            label_path,
            [_label_row(market_id, decision_ts, outcome, action) for action in ACTIONS],
        )
        _write_jsonl(
            metadata_path,
            [
                {
                    "market_id": market_id,
                    "condition_id": market_id,
                    "slug": f"btc-updown-5m-{index}",
                    "market_start_ts": decision_ts - 60_000,
                    "market_end_ts": decision_ts + 180_000,
                    "paper_only": True,
                    "capital_at_risk": False,
                }
            ],
        )
        _write_jsonl(
            resolution_path,
            [
                {
                    "market_id": market_id,
                    "resolved_outcome": outcome,
                    "paper_only": True,
                    "capital_at_risk": False,
                }
            ],
        )
        manifest_path = corpus_dir / "polymarket_corpus_manifest.json"
        _write_json(
            manifest_path,
            {
                "schema_version": "bigan-v8-polymarket-corpus-v3",
                "normalized_artifact_hashes": {
                    "feature_rows": _sha256(feature_path),
                    "label_rows": _sha256(label_path),
                    "market_metadata": _sha256(metadata_path),
                    "resolution_events": _sha256(resolution_path),
                },
                "paper_only": True,
                "capital_at_risk": False,
            },
        )
        role_rows.append(
            {
                "selection_rank": index + 1,
                "role": role,
                "market_id": market_id,
                "source_corpus_dir": str(corpus_dir),
                "corpus_manifest": {
                    "path": str(manifest_path.resolve()),
                    "sha256": _sha256(manifest_path),
                },
                "labels_or_outcomes_opened_for_role_assignment": False,
            }
        )
    role_rows_path = tmp_path / "role_rows.jsonl"
    _write_jsonl(role_rows_path, role_rows)
    exclusion_registry_path = tmp_path / "prior_evidence_exclusion_registry.json"
    _write_json(
        exclusion_registry_path,
        {
            "prior_market_ids": [f"prior-market-{index:03d}" for index in range(95)],
            "paper_only": True,
            "capital_at_risk": False,
        },
    )
    manifest_path = tmp_path / "role_assignment_manifest.json"
    _write_json(
        manifest_path,
        {
            "role_assignment_ready": True,
            "labels_or_outcomes_opened_for_role_assignment": False,
            "protocol": {
                "path": str(PROTOCOL_PATH.resolve()),
                "sha256": _sha256(PROTOCOL_PATH),
            },
            "feature_contract": {
                "path": str(FEATURE_CONTRACT_PATH.resolve()),
                "sha256": _sha256(FEATURE_CONTRACT_PATH),
            },
            "prior_evidence_exclusion_registry": {
                "path": str(exclusion_registry_path.resolve()),
                "sha256": _sha256(exclusion_registry_path),
            },
            "selected_rows": {
                "path": str(role_rows_path.resolve()),
                "sha256": _sha256(role_rows_path),
            },
            "paper_only": True,
            "capital_at_risk": False,
        },
    )
    return manifest_path


def _feature_row(market_id: str, decision_ts: int, outcome: str) -> dict:
    up_regime = outcome == "UP"
    up_bid, up_ask = (0.69, 0.71) if up_regime else (0.29, 0.31)
    down_bid, down_ask = (0.29, 0.31) if up_regime else (0.69, 0.71)
    signal = 0.01 if up_regime else -0.01
    features = {
        "btc_return_10s": signal,
        "btc_return_30s": signal,
        "btc_return_1m": signal,
        "btc_return_5m": signal,
        "btc_return_15m": signal,
        "btc_volatility_1m": 0.01,
        "btc_volatility_5m": 0.02,
        "btc_volatility_15m": 0.03,
        "reference_price_to_beat_distance_at_decision": signal,
        "time_to_close_seconds": 180.0,
        "market_age_seconds": 60.0,
        "combined_spread_bps": 500.0,
        "liquidity_imbalance": signal,
        "recent_up_trade_volume": 10.0 if up_regime else 2.0,
        "recent_down_trade_volume": 2.0 if up_regime else 10.0,
        "up_bid": up_bid,
        "up_ask": up_ask,
        "up_mid": (up_bid + up_ask) / 2.0,
        "down_bid": down_bid,
        "down_ask": down_ask,
        "down_mid": (down_bid + down_ask) / 2.0,
    }
    for side in ("up", "down"):
        features.update(
            {
                f"{side}_spread_bps": 500.0,
                f"{side}_queue_fill_probability_proxy": 0.9,
                f"{side}_book_staleness_ms": 100.0,
                f"{side}_liquidity_depth": 100.0,
                f"{side}_executable_ask_notional": 50.0,
                f"{side}_executable_bid_notional": 50.0,
                f"{side}_recent_book_update_count_1m": 10.0,
                f"{side}_recent_spread_stability_1m": 0.9,
                f"{side}_recent_bid_depth_volatility_1m": 0.1,
            }
        )
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "max_input_ts": decision_ts - 1,
        "features": features,
        "feature_provenance": {
            "reference_price_to_beat_distance_at_decision": {
                "max_input_ts": decision_ts - 1,
                "decision_ts": decision_ts,
                "provenance_valid": True,
                "source": "synthetic_chainlink_test_fixture",
            }
        },
        "paper_only": True,
        "capital_at_risk": False,
    }


def _label_row(
    market_id: str,
    decision_ts: int,
    outcome: str,
    action: str,
) -> dict:
    if action == "NO_TRADE":
        target = 0.0
    else:
        side = "UP" if "BUY_UP" in action else "DOWN"
        correct = side == outcome
        if "HOLD_TO_SETTLEMENT" in action:
            target = 0.40 if correct else -0.20
        else:
            target = 0.25 if correct else -0.10
    return {
        "market_id": market_id,
        "decision_ts": decision_ts,
        "action": action,
        "total_net_pnl_per_notional": target,
        "fees": 0.001 if action != "NO_TRADE" else 0.0,
        "slippage": 0.002 if action != "NO_TRADE" else 0.0,
        "liquidity_impact": 0.001 if action != "NO_TRADE" else 0.0,
        "paper_only": True,
        "capital_at_risk": False,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
