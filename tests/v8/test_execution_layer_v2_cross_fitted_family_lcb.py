from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb import (
    CrossFittedFamilyLCBPrecollectionFreezeConfig,
    CrossFittedFamilyLCBRoleAssignmentConfig,
    assign_cross_fitted_family_lcb_roles,
    freeze_cross_fitted_family_lcb_precollection,
    validate_cross_fitted_family_lcb_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_feature_contract_v1.json"
)


def test_cross_fitted_family_lcb_protocol_is_frozen_and_safe() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_cross_fitted_family_lcb_protocol(protocol)
    assert protocol["role_assignment"]["target_valid_market_count"] == 90
    assert protocol["cross_fit_protocol"]["fold_count"] == 5
    assert protocol["conformal_lcb_protocol"]["affine_calibration_enabled"] is False
    assert protocol["collector_contract"]["public_provider_timeout_seconds"] == 330.0
    assert protocol["collector_contract"]["public_provider_http_timeout_seconds"] == 5.0
    assert protocol["uses_prior_validation_or_future_labels_for_tuning"] is False
    assert protocol["safety"]["v8_execution_handoff_allowed"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["role_assignment"]["development_train_market_count"] = 39
    with pytest.raises(ValueError, match="role_total"):
        validate_cross_fitted_family_lcb_protocol(drifted)

    short_window = json.loads(json.dumps(protocol))
    short_window["collector_contract"]["public_provider_timeout_seconds"] = 20.0
    with pytest.raises(ValueError, match="full_round_ws_collection_window"):
        validate_cross_fitted_family_lcb_protocol(short_window)


def test_freeze_precollection_roles_and_prior_market_exclusions(tmp_path: Path) -> None:
    historical_registry = tmp_path / "historical_split_manifest.json"
    historical_markets = [f"historical-{index:03d}" for index in range(65)]
    _write_json(
        historical_registry,
        {
            "split_summary": {
                "splits": {
                    "historical_train": {
                        "market_ids": historical_markets[:39],
                        "minimum_decision_ts": 1_000_000,
                        "maximum_decision_ts": 4_000_000,
                    },
                    "historical_calibration": {
                        "market_ids": historical_markets[39:52],
                        "minimum_decision_ts": 4_300_000,
                        "maximum_decision_ts": 5_500_000,
                    },
                    "historical_validation": {
                        "market_ids": historical_markets[52:],
                        "minimum_decision_ts": 5_800_000,
                        "maximum_decision_ts": 7_000_000,
                    },
                }
            }
        },
    )
    future_registry = tmp_path / "future_decisions.jsonl"
    _write_jsonl(
        future_registry,
        [
            {
                "market_id": f"future-{index:03d}",
                "decision_ts": 8_000_000 + index * 300_000,
                "row_identity": f"future-row-{index}",
            }
            for index in range(30)
        ],
    )
    evidence = tmp_path / "rejected_candidate_manifest.json"
    _write_json(evidence, {"candidate_frozen_for_future_evaluation": False})

    result = freeze_cross_fitted_family_lcb_precollection(
        CrossFittedFamilyLCBPrecollectionFreezeConfig(
            run_id="issue172-precollection",
            output_dir=tmp_path / "runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            git_commit="a" * 40,
            prior_market_registry_pins=(
                (historical_registry, _sha256(historical_registry)),
                (future_registry, _sha256(future_registry)),
            ),
            prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
            expected_prior_unique_market_count=95,
        )
    )

    manifest = result["manifest"]
    exclusion = json.loads(
        Path(manifest["prior_evidence_exclusion_registry"]["path"]).read_text()
    )
    assert exclusion["prior_unique_market_count"] == 95
    assert exclusion["prior_outcome_or_pnl_values_loaded"] is False
    assert exclusion["prior_validation_or_future_evidence_used_for_tuning"] is False
    assert manifest["role_plan"] == [
        {
            "role": "development_train",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": 40,
        },
        {
            "role": "development_calibration",
            "valid_market_rank_start": 41,
            "valid_market_rank_end": 60,
        },
        {
            "role": "confirmatory_validation",
            "valid_market_rank_start": 61,
            "valid_market_rank_end": 90,
        },
    ]
    assert manifest["minimum_collection_decision_ts"] > 16_700_000
    assert manifest["role_assignment_outcome_blind"] is True
    assert manifest["feature_contract"] == {
        "path": str(FEATURE_CONTRACT_PATH.resolve()),
        "sha256": _sha256(FEATURE_CONTRACT_PATH),
    }
    assert manifest["collection_started"] is False
    assert manifest["model_fit_started"] is False
    assert manifest["source_model_candidate_eligible"] is False
    assert manifest["freeze_ready"] is False
    assert manifest["promotion_evidence_eligible"] is False
    assert manifest["v8_execution_handoff_allowed"] is False
    assert manifest["#134_resume_allowed"] is False
    assert manifest["#146_start_allowed"] is False
    assert manifest["paper_only"] is True
    assert manifest["capital_at_risk"] is False


def test_precollection_freeze_rejects_outcome_registry_and_hash_tamper(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "market_ids": [f"market-{index}" for index in range(95)],
            "maximum_decision_ts": 10_000_000,
            "net_pnl": 1.0,
        },
    )
    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"status": "blocked"})
    config = CrossFittedFamilyLCBPrecollectionFreezeConfig(
        run_id="leaky-registry",
        output_dir=tmp_path / "runs",
        protocol_path=PROTOCOL_PATH,
        expected_protocol_sha256=_sha256(PROTOCOL_PATH),
        feature_contract_path=FEATURE_CONTRACT_PATH,
        expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
        git_commit="b" * 40,
        prior_market_registry_pins=((registry, _sha256(registry)),),
        prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
        expected_prior_unique_market_count=95,
    )
    with pytest.raises(ValueError, match="forbidden outcome fields"):
        freeze_cross_fitted_family_lcb_precollection(config)

    clean_registry = tmp_path / "clean_registry.json"
    _write_json(
        clean_registry,
        {
            "market_ids": [f"market-{index}" for index in range(95)],
            "maximum_decision_ts": 10_000_000,
        },
    )
    tampered = CrossFittedFamilyLCBPrecollectionFreezeConfig(
        run_id="tampered-registry",
        output_dir=tmp_path / "runs",
        protocol_path=PROTOCOL_PATH,
        expected_protocol_sha256=_sha256(PROTOCOL_PATH),
        feature_contract_path=FEATURE_CONTRACT_PATH,
        expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
        git_commit="c" * 40,
        prior_market_registry_pins=((clean_registry, "0" * 64),),
        prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
        expected_prior_unique_market_count=95,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze_cross_fitted_family_lcb_precollection(tampered)


def test_assigns_frozen_roles_without_opening_labels_or_outcomes(
    tmp_path: Path,
) -> None:
    fixture = _role_assignment_fixture(tmp_path, market_count=90)
    result = assign_cross_fitted_family_lcb_roles(
        CrossFittedFamilyLCBRoleAssignmentConfig(
            run_id="issue172-role-assignment",
            output_dir=tmp_path / "role-runs",
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=_sha256(
                fixture["freeze_path"]
            ),
            batch_progress_pins=(
                (fixture["batch_path"], _sha256(fixture["batch_path"])),
            ),
            training_corpus_root=fixture["training_root"],
        )
    )

    report = result["report"]
    assert report["status"] == "OUTCOME_BLIND_ROLE_ASSIGNMENT_READY"
    assert report["role_assignment_ready"] is True
    assert report["selected_market_count"] == 90
    assert report["role_market_counts"] == {
        "confirmatory_validation": 30,
        "development_calibration": 20,
        "development_train": 40,
    }
    assert report["role_assignment_uses_outcomes"] is False
    assert report["labels_or_outcomes_opened_for_role_assignment"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert result["manifest"]["feature_contract"] == {
        "path": str(FEATURE_CONTRACT_PATH.resolve()),
        "sha256": _sha256(FEATURE_CONTRACT_PATH),
    }
    assert result["manifest_path"].name == "role_assignment_manifest.json"
    rows = [
        json.loads(line)
        for line in result["selected_rows_path"].read_text().splitlines()
    ]
    assert [row["role"] for row in rows[:40]] == ["development_train"] * 40
    assert [row["role"] for row in rows[40:60]] == [
        "development_calibration"
    ] * 20
    assert [row["role"] for row in rows[60:]] == [
        "confirmatory_validation"
    ] * 30
    assert all(
        row["labels_or_outcomes_opened_for_role_assignment"] is False
        for row in rows
    )


def test_role_assignment_fails_closed_when_chainlink_capture_is_missing(
    tmp_path: Path,
) -> None:
    fixture = _role_assignment_fixture(
        tmp_path,
        market_count=90,
        missing_chainlink_index=17,
    )
    result = assign_cross_fitted_family_lcb_roles(
        CrossFittedFamilyLCBRoleAssignmentConfig(
            run_id="issue172-chainlink-gap",
            output_dir=tmp_path / "role-runs",
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=_sha256(
                fixture["freeze_path"]
            ),
            batch_progress_pins=(
                (fixture["batch_path"], _sha256(fixture["batch_path"])),
            ),
            training_corpus_root=fixture["training_root"],
        )
    )

    report = result["report"]
    assert report["role_assignment_ready"] is False
    assert report["selected_market_count"] == 89
    assert "insufficient_quality_valid_unique_market_support" in report[
        "blocking_reason_codes"
    ]
    assert report["excluded_reason_distribution"] == {
        "chainlink_rtds_coverage_failed": 1
    }
    assert report["source_model_candidate_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False


def test_role_assignment_excludes_prior_market_and_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _role_assignment_fixture(
        tmp_path,
        market_count=90,
        prior_overlap_index=8,
    )
    result = assign_cross_fitted_family_lcb_roles(
        CrossFittedFamilyLCBRoleAssignmentConfig(
            run_id="issue172-prior-overlap",
            output_dir=tmp_path / "role-runs",
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=_sha256(
                fixture["freeze_path"]
            ),
            batch_progress_pins=(
                (fixture["batch_path"], _sha256(fixture["batch_path"])),
            ),
            training_corpus_root=fixture["training_root"],
        )
    )

    report = result["report"]
    assert report["role_assignment_ready"] is False
    assert report["selected_market_count"] == 89
    assert report["prior_market_overlap_count"] == 0
    assert report["excluded_reason_distribution"] == {
        "feature_market_overlaps_prior_evidence": 1
    }
    assert report["source_model_candidate_eligible"] is False


def test_role_assignment_rejects_chainlink_capture_with_proxy_reference_feature(
    tmp_path: Path,
) -> None:
    fixture = _role_assignment_fixture(
        tmp_path,
        market_count=90,
        proxy_reference_index=23,
    )
    result = assign_cross_fitted_family_lcb_roles(
        CrossFittedFamilyLCBRoleAssignmentConfig(
            run_id="issue172-chainlink-proxy-reference",
            output_dir=tmp_path / "role-runs",
            precollection_freeze_manifest_path=fixture["freeze_path"],
            expected_precollection_freeze_manifest_sha256=_sha256(
                fixture["freeze_path"]
            ),
            batch_progress_pins=(
                (fixture["batch_path"], _sha256(fixture["batch_path"])),
            ),
            training_corpus_root=fixture["training_root"],
        )
    )

    report = result["report"]
    assert report["role_assignment_ready"] is False
    assert report["selected_market_count"] == 89
    assert report["excluded_reason_distribution"][
        "feature_reference_distance_not_chainlink_sourced"
    ] == 1
    assert report["source_model_candidate_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False


def _role_assignment_fixture(
    tmp_path: Path,
    *,
    market_count: int,
    missing_chainlink_index: int | None = None,
    prior_overlap_index: int | None = None,
    proxy_reference_index: int | None = None,
) -> dict[str, Path]:
    registry = tmp_path / "prior_registry.json"
    _write_json(
        registry,
        {
            "market_ids": [f"prior-{index:03d}" for index in range(95)],
            "maximum_decision_ts": 10_000_000,
        },
    )
    evidence = tmp_path / "prior_evidence.json"
    _write_json(evidence, {"status": "rejected"})
    freeze = freeze_cross_fitted_family_lcb_precollection(
        CrossFittedFamilyLCBPrecollectionFreezeConfig(
            run_id="issue172-freeze",
            output_dir=tmp_path / "freeze-runs",
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=_sha256(PROTOCOL_PATH),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(FEATURE_CONTRACT_PATH),
            git_commit="d" * 40,
            prior_market_registry_pins=((registry, _sha256(registry)),),
            prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
            expected_prior_unique_market_count=95,
        )
    )
    freeze_path = freeze["manifest_path"]
    minimum_decision_ts = int(freeze["manifest"]["minimum_collection_decision_ts"])
    training_root = tmp_path / "training"
    captures = []
    finalizations = []
    for index in range(market_count):
        run_id = f"capture-{index + 1:03d}"
        market_id = (
            "prior-000"
            if index == prior_overlap_index
            else f"new-market-{index + 1:03d}"
        )
        decision_ts = minimum_decision_ts + (index + 1) * 300_000
        corpus_dir = training_root / "polymarket" / market_id
        corpus_dir.mkdir(parents=True)
        feature_path = corpus_dir / "polymarket_feature_rows.jsonl"
        reference_source = (
            "market_start_reference_candle_open_price"
            if index == proxy_reference_index
            else "polymarket_rtds_chainlink_market_start"
        )
        _write_jsonl(
            feature_path,
            [
                {
                    "market_id": market_id,
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts - 1,
                    "features": {
                        "up_bid": 0.45,
                        "up_ask": 0.46,
                        "reference_price_to_beat_distance_at_decision": 0.001,
                    },
                    "feature_provenance": {
                        "reference_price_to_beat_distance_at_decision": {
                            "source": "polymarket_corpus",
                            "source_fields_used": (
                                "polymarket_btc_reference_candles."
                                "open_price_at_market_start|"
                                "polymarket_btc_reference_candles."
                                "close_price_at_decision"
                                if index == proxy_reference_index
                                else "raw_polymarket_chainlink_prices."
                                "price_at_or_before_market_start|"
                                "raw_polymarket_chainlink_prices."
                                "price_at_or_before_decision"
                            ),
                            "max_input_ts": decision_ts - 1,
                            "available_at_ts": decision_ts - 1,
                            "decision_ts": decision_ts,
                            "provenance_valid": True,
                            "reference_price_to_beat_source": reference_source,
                        }
                    },
                }
            ],
        )
        chainlink_path = corpus_dir / "polymarket_chainlink_prices.jsonl"
        _write_jsonl(
            chainlink_path,
            [
                {
                    "source_ts": decision_ts - 1,
                    "available_at_ts": decision_ts - 1,
                    "price": 65_000.0,
                    "source_type": "polymarket_rtds_chainlink",
                    "symbol": "btc/usd",
                    "read_only": True,
                    "paper_only": True,
                    "capital_at_risk": False,
                }
            ],
        )
        chainlink_manifest_path = (
            corpus_dir / "polymarket_chainlink_decision_time_evidence_manifest.json"
        )
        chainlink_integration_passed = index != proxy_reference_index
        chainlink_manifest = {
            "schema_version": (
                "bigan-v8-polymarket-chainlink-decision-time-evidence-v2"
            ),
            "source_type": "polymarket_rtds_chainlink",
            "decision_time_only": True,
            "row_count": 1,
            "evidence_path": chainlink_path.name,
            "evidence_sha256": _sha256(chainlink_path),
            "feature_row_count": 1,
            "integrated_feature_row_count": int(chainlink_integration_passed),
            "missing_or_invalid_feature_row_count": int(
                not chainlink_integration_passed
            ),
            "feature_reference_source_distribution": {reference_source: 1},
            "feature_integration_reason_distribution": (
                {}
                if chainlink_integration_passed
                else {"reference_distance_not_sourced_from_chainlink": 1}
            ),
            "timestamp_causality_violation_count": 0,
            "feature_builder_integration_passed": chainlink_integration_passed,
            "feature_builder_integration_required": not chainlink_integration_passed,
            "read_only": True,
            "paper_only": True,
            "capital_at_risk": False,
            "broker_exchange_write_enabled": False,
            "live_exchange_write_enabled": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        }
        _write_json(chainlink_manifest_path, chainlink_manifest)
        _write_json(
            corpus_dir / "polymarket_corpus_manifest.json",
            {
                "schema_version": "bigan-v8-polymarket-corpus-v3",
                "normalized_artifact_hashes": {
                    "feature_rows": _sha256(feature_path),
                    "chainlink_prices": _sha256(chainlink_path),
                    "chainlink_decision_time_evidence_manifest": _sha256(
                        chainlink_manifest_path
                    ),
                },
                "chainlink_decision_time_feature_integration": chainlink_manifest,
                "paper_only": True,
                "capital_at_risk": False,
            },
        )
        _write_json(
            corpus_dir / "training_corpus_provenance.json",
            {
                "chainlink_decision_time_evidence": {
                    "attached": chainlink_integration_passed,
                    "row_count": 1,
                    "evidence_filename": chainlink_path.name,
                    "evidence_sha256": _sha256(chainlink_path),
                    "manifest_filename": chainlink_manifest_path.name,
                    "manifest_sha256": _sha256(chainlink_manifest_path),
                    "feature_builder_integration_passed": (
                        chainlink_integration_passed
                    ),
                    "feature_builder_integration_required": (
                        not chainlink_integration_passed
                    ),
                }
            },
        )
        # This must remain unreadable by the outcome-blind role assignment stage.
        (corpus_dir / "polymarket_label_rows.jsonl").write_text(
            "not-json-and-must-not-be-opened\n",
            encoding="utf-8",
        )
        captures.append(
            {
                "run_id": run_id,
                "round_index": index + 1,
                "scheduled_round_start_ts": decision_ts - 60_000,
                "capture_start_boundary_validation_passed": True,
                "capture_status": "pending_resolution",
                "raw_polymarket_market_count": 1,
                "provider_raw_orderbook_snapshot_count": 8,
                "training_sampled_orderbook_row_count": 4,
                "raw_btc_candle_row_count": 12,
                "raw_chainlink_price_row_count": (
                    0 if index == missing_chainlink_index else 20
                ),
                "reject_reason_counts": {},
            }
        )
        finalizations.append(
            {
                "run_id": run_id,
                "finalization_status": "exported",
                "pending_resolution": False,
                "training_eligible": True,
                "raw_resolution_count": 1,
                "reject_reason_counts": {},
                "exported_training_corpus_dir": str(corpus_dir),
            }
        )
    batch_path = tmp_path / "batch_progress.json"
    _write_json(
        batch_path,
        {
            "batch_id": "issue172-batch01",
            "capture_count": len(captures),
            "error_count": 0,
            "exported_round_count": len(finalizations),
            "captures": captures,
            "finalizations": finalizations,
            "errors": [],
            "paper_only": True,
            "capital_at_risk": False,
        },
    )
    return {
        "freeze_path": freeze_path,
        "batch_path": batch_path,
        "training_root": training_root,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
