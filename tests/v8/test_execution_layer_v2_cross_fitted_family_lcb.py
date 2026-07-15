from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_cross_fitted_family_lcb import (
    CrossFittedFamilyLCBPrecollectionFreezeConfig,
    freeze_cross_fitted_family_lcb_precollection,
    validate_cross_fitted_family_lcb_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_cross_fitted_family_lcb_v1.json"
)


def test_cross_fitted_family_lcb_protocol_is_frozen_and_safe() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_cross_fitted_family_lcb_protocol(protocol)
    assert protocol["role_assignment"]["target_valid_market_count"] == 90
    assert protocol["cross_fit_protocol"]["fold_count"] == 5
    assert protocol["conformal_lcb_protocol"]["affine_calibration_enabled"] is False
    assert protocol["uses_prior_validation_or_future_labels_for_tuning"] is False
    assert protocol["safety"]["v8_execution_handoff_allowed"] is False

    drifted = json.loads(json.dumps(protocol))
    drifted["role_assignment"]["development_train_market_count"] = 39
    with pytest.raises(ValueError, match="role_total"):
        validate_cross_fitted_family_lcb_protocol(drifted)


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
        git_commit="c" * 40,
        prior_market_registry_pins=((clean_registry, "0" * 64),),
        prior_evidence_artifact_pins=((evidence, _sha256(evidence)),),
        expected_prior_unique_market_count=95,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        freeze_cross_fitted_family_lcb_precollection(tampered)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
