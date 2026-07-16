"""Tests for the hybrid pairwise fresh precollection readiness gate."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.hybrid_pairwise_precollection_readiness import (
    HybridPairwisePrecollectionReadinessConfig,
    evaluate_hybrid_pairwise_precollection_readiness,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_v1.json"
)
FEATURE_CONTRACT_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
HYBRID_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_hybrid_pairwise_fresh_calibration_v1.json"
)


def test_committed_protocol_freezes_roles_and_source_contracts() -> None:
    protocol = _load_json(HYBRID_PROTOCOL_PATH)
    source = _load_json(SOURCE_PROTOCOL_PATH)

    assert protocol["fresh_role_plan"] == [
        {
            "role": "fresh_development_calibration",
            "valid_market_rank_start": 1,
            "valid_market_rank_end": 45,
        },
        {
            "role": "fresh_confirmatory_validation",
            "valid_market_rank_start": 46,
            "valid_market_rank_end": 105,
        },
    ]
    assert protocol["collection_plan"]["target_valid_unique_market_count"] == 105
    assert protocol["collection_plan"]["initial_capture_attempt_count"] == 120
    assert protocol["collection_plan"]["maximum_total_capture_attempt_count"] == 150
    expected_hashes = {
        "collector_contract_sha256": canonical_json_sha256(
            source["collector_contract"]
        ),
        "action_advantage_lcb_protocol_sha256": canonical_json_sha256(
            source["action_advantage_lcb_protocol"]
        ),
        "development_freeze_gates_sha256": canonical_json_sha256(
            source["development_freeze_gates"]
        ),
        "confirmatory_validation_gates_sha256": canonical_json_sha256(
            source["confirmatory_validation_gates"]
        ),
        "frozen_execution_contract_sha256": canonical_json_sha256(
            source["frozen_execution_contract"]
        ),
    }
    assert protocol["source_contract_hashes"] == expected_hashes
    assert protocol["calibration_contract"]["ranker_retraining_allowed"] is False
    assert (
        protocol["calibration_contract"]["ranker_score_mutation_allowed"] is False
    )
    assert (
        protocol["calibration_contract"][
            "uses_current_oof_or_validation_metrics_for_tuning"
        ]
        is False
    )


def test_active_lineage_and_missing_quarantine_block_before_freeze(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    active_state = tmp_path / "active.json"
    _write_json(
        active_state,
        {
            "status": "waiting_for_issue177_batch01",
            "capture_count": 33,
            "exported_round_count": 23,
            "pending_resolution_count": 1,
            "error_count": 0,
        },
    )

    result = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_state,),
        quarantine_path=None,
    )
    report = result["report"]

    assert report["readiness_status"] == "blocked_fail_closed"
    assert report["precollection_readiness_passed"] is False
    assert report["precollection_freeze_created"] is False
    assert result["freeze_manifest_path"] is None
    assert "active_prior_lineage_incomplete" in report["blocking_reason_codes"]
    assert (
        "final_prior_lineage_quarantine_missing"
        in report["blocking_reason_codes"]
    )
    _assert_blocked_safety(report)


def test_complete_lineage_and_valid_quarantine_create_freeze_only(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    active_state = tmp_path / "active.json"
    quarantine_path = tmp_path / "quarantine.json"
    _write_json(active_state, {"status": "prior_lineage_complete"})
    _write_json(quarantine_path, _valid_quarantine())

    result = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_state,),
        quarantine_path=quarantine_path,
    )
    report = result["report"]
    freeze = result["freeze_manifest"]

    assert report["readiness_status"] == "ready_for_separate_future_collection_freeze"
    assert report["precollection_readiness_passed"] is True
    assert report["precollection_freeze_created"] is True
    assert freeze is not None
    assert freeze["minimum_collection_decision_ts"] == 3_001
    assert freeze["fresh_role_plan"][0]["valid_market_rank_end"] == 45
    assert freeze["fresh_role_plan"][1]["valid_market_rank_end"] == 105
    assert freeze["collection_plan"]["initial_capture_attempt_count"] == 120
    assert freeze["collection_plan"]["maximum_total_capture_attempt_count"] == 150
    assert freeze["collection_started"] is False
    assert freeze["collection_start_allowed"] is False
    assert freeze["collection_start_command_generated"] is False
    assert freeze["ranker_retraining_allowed"] is False
    assert freeze["ranker_score_mutation_allowed"] is False
    _assert_blocked_safety(report)
    _assert_blocked_safety(freeze)


def test_overwrite_removes_stale_ready_freeze_when_gate_becomes_blocked(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    active_ready = tmp_path / "active-ready.json"
    active_blocked = tmp_path / "active-blocked.json"
    quarantine_path = tmp_path / "quarantine.json"
    _write_json(active_ready, {"status": "prior_lineage_complete"})
    _write_json(active_blocked, {"status": "waiting_for_issue177_batch01"})
    _write_json(quarantine_path, _valid_quarantine())

    ready = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_ready,),
        quarantine_path=quarantine_path,
        run_id="same-run",
    )
    assert ready["freeze_manifest_path"].is_file()

    blocked = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_blocked,),
        quarantine_path=None,
        run_id="same-run",
        overwrite=True,
    )
    assert blocked["freeze_manifest_path"] is None
    assert not (
        Path(blocked["run_dir"])
        / "hybrid_pairwise_precollection_freeze_manifest.json"
    ).exists()


def test_quarantine_hash_drift_fails_closed_before_evaluation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    active_state = tmp_path / "active.json"
    quarantine_path = tmp_path / "quarantine.json"
    _write_json(active_state, {"status": "prior_lineage_complete"})
    _write_json(quarantine_path, _valid_quarantine())
    quarantine_hash = _sha256(quarantine_path)
    _write_json(quarantine_path, {**_valid_quarantine(), "tampered": True})

    with pytest.raises(ValueError, match="final prior quarantine SHA-256 mismatch"):
        _run(
            tmp_path,
            fixture,
            active_state_paths=(active_state,),
            quarantine_path=quarantine_path,
            quarantine_sha256=quarantine_hash,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            {"settlement_pnl": 1.0},
            "final_prior_quarantine_outcome_blind_failed",
        ),
        (
            {"maximum_prior_decision_ts": 3_001},
            "final_prior_quarantine_chronology_failed",
        ),
        (
            {
                "historical_development_market_ids_sha256": "f" * 64,
            },
            "final_prior_quarantine_historical_training_markets_quarantined_failed",
        ),
    ],
)
def test_invalid_quarantine_evidence_blocks_freeze(
    tmp_path: Path,
    mutation: dict[str, Any],
    reason: str,
) -> None:
    fixture = _fixture(tmp_path)
    active_state = tmp_path / "active.json"
    quarantine_path = tmp_path / "quarantine.json"
    _write_json(active_state, {"status": "prior_lineage_complete"})
    _write_json(quarantine_path, {**_valid_quarantine(), **mutation})

    result = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_state,),
        quarantine_path=quarantine_path,
    )

    assert result["report"]["precollection_readiness_passed"] is False
    assert reason in result["report"]["blocking_reason_codes"]
    assert result["freeze_manifest_path"] is None


def test_protocol_role_or_ranker_identity_drift_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    protocol = _load_json(fixture["hybrid_protocol_path"])
    protocol["fresh_role_plan"][1]["valid_market_rank_start"] = 45
    _write_json(fixture["hybrid_protocol_path"], protocol)
    fixture["hybrid_protocol_sha256"] = _sha256(fixture["hybrid_protocol_path"])
    active_state = tmp_path / "active.json"
    _write_json(active_state, {"status": "prior_lineage_complete"})

    with pytest.raises(ValueError, match="invalid hybrid pairwise protocol: roles"):
        _run(
            tmp_path,
            fixture,
            active_state_paths=(active_state,),
            quarantine_path=None,
        )


def test_active_state_forbidden_evidence_cannot_be_marked_complete(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    active_state = tmp_path / "active.json"
    quarantine_path = tmp_path / "quarantine.json"
    _write_json(
        active_state,
        {
            "status": "prior_lineage_complete",
            "realized_pnl": 0.5,
        },
    )
    _write_json(quarantine_path, _valid_quarantine())

    result = _run(
        tmp_path,
        fixture,
        active_state_paths=(active_state,),
        quarantine_path=quarantine_path,
    )

    snapshot = result["report"]["active_lineage_snapshots"][0]
    assert snapshot["lineage_complete"] is False
    assert snapshot["forbidden_field_paths"] == ["realized_pnl"]
    assert "active_prior_lineage_incomplete" in result["report"][
        "blocking_reason_codes"
    ]


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source_protocol = _load_json(SOURCE_PROTOCOL_PATH)
    protocol = deepcopy(_load_json(HYBRID_PROTOCOL_PATH))

    ranker_manifest = {
        "schema_version": "fixture-ranker-manifest-v1",
        "freeze_id": protocol["historical_ranker_freeze"]["freeze_id"],
        "model_sha256": protocol["historical_ranker_freeze"]["model_sha256"],
        "dataset_hash": protocol["historical_ranker_freeze"]["dataset_hash"],
        "oof_dataset_hash": protocol["historical_ranker_freeze"][
            "oof_dataset_hash"
        ],
        "split_hash": protocol["historical_ranker_freeze"]["split_hash"],
        "model_config_hash": protocol["historical_ranker_freeze"][
            "model_config_hash"
        ],
        "rank_scores_execution_eligible": False,
    }
    ranker_manifest_path = tmp_path / "ranker-manifest.json"
    _write_json(ranker_manifest_path, ranker_manifest)
    ranker_descriptor = {
        "schema_version": "fixture-ranker-descriptor-v1",
        "freeze_id": ranker_manifest["freeze_id"],
        "model_sha256": ranker_manifest["model_sha256"],
        "model": {"path": "fixture-model.json", "sha256": ranker_manifest["model_sha256"]},
        "dataset_hash": ranker_manifest["dataset_hash"],
        "split_hash": ranker_manifest["split_hash"],
        "model_config_hash": ranker_manifest["model_config_hash"],
        "freeze_manifest": {
            "path": str(ranker_manifest_path),
            "sha256": _sha256(ranker_manifest_path),
        },
        "fresh_calibration_required": True,
        "rank_scores_execution_eligible": False,
    }
    ranker_descriptor_path = tmp_path / "ranker-descriptor.json"
    _write_json(ranker_descriptor_path, ranker_descriptor)

    registry_descriptor = {
        "schema_version": "fixture-registry-descriptor-v1",
        "selected_market_count": 90,
        "selected_market_ids_sha256": protocol["historical_development_registry"][
            "selected_market_ids_sha256"
        ],
    }
    registry_descriptor_path = tmp_path / "registry-descriptor.json"
    _write_json(registry_descriptor_path, registry_descriptor)

    protocol["source_pairwise_protocol_sha256"] = _sha256(SOURCE_PROTOCOL_PATH)
    protocol["source_feature_contract_sha256"] = _sha256(FEATURE_CONTRACT_PATH)
    protocol["historical_development_registry"]["descriptor_sha256"] = _sha256(
        registry_descriptor_path
    )
    protocol["historical_ranker_freeze"]["descriptor_sha256"] = _sha256(
        ranker_descriptor_path
    )
    protocol["source_contract_hashes"] = {
        "collector_contract_sha256": canonical_json_sha256(
            source_protocol["collector_contract"]
        ),
        "action_advantage_lcb_protocol_sha256": canonical_json_sha256(
            source_protocol["action_advantage_lcb_protocol"]
        ),
        "development_freeze_gates_sha256": canonical_json_sha256(
            source_protocol["development_freeze_gates"]
        ),
        "confirmatory_validation_gates_sha256": canonical_json_sha256(
            source_protocol["confirmatory_validation_gates"]
        ),
        "frozen_execution_contract_sha256": canonical_json_sha256(
            source_protocol["frozen_execution_contract"]
        ),
    }
    hybrid_protocol_path = tmp_path / "hybrid-protocol.json"
    _write_json(hybrid_protocol_path, protocol)
    return {
        "hybrid_protocol_path": hybrid_protocol_path,
        "hybrid_protocol_sha256": _sha256(hybrid_protocol_path),
        "source_protocol_sha256": _sha256(SOURCE_PROTOCOL_PATH),
        "feature_contract_sha256": _sha256(FEATURE_CONTRACT_PATH),
        "registry_descriptor_path": registry_descriptor_path,
        "registry_descriptor_sha256": _sha256(registry_descriptor_path),
        "ranker_descriptor_path": ranker_descriptor_path,
        "ranker_descriptor_sha256": _sha256(ranker_descriptor_path),
        "ranker_manifest_path": ranker_manifest_path,
        "ranker_manifest_sha256": _sha256(ranker_manifest_path),
    }


def _valid_quarantine() -> dict[str, Any]:
    protocol = _load_json(HYBRID_PROTOCOL_PATH)
    return {
        "schema_version": "fixture-final-prior-lineage-quarantine-v1",
        "final": True,
        "active_prior_lineage_complete": True,
        "includes_issue175_through_issue179": True,
        "historical_development_market_ids_sha256": protocol[
            "historical_development_registry"
        ]["selected_market_ids_sha256"],
        "maximum_prior_decision_ts": 2_000,
        "outcome_label_or_pnl_artifacts_opened": False,
        "resolution_artifacts_opened": False,
        "safety": {
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
        },
    }


def _run(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    active_state_paths: tuple[Path, ...],
    quarantine_path: Path | None,
    quarantine_sha256: str | None = None,
    run_id: str = "readiness",
    overwrite: bool = False,
) -> dict[str, Any]:
    return evaluate_hybrid_pairwise_precollection_readiness(
        HybridPairwisePrecollectionReadinessConfig(
            run_id=run_id,
            output_dir=tmp_path / "runs",
            hybrid_protocol_path=fixture["hybrid_protocol_path"],
            expected_hybrid_protocol_sha256=fixture["hybrid_protocol_sha256"],
            source_pairwise_protocol_path=SOURCE_PROTOCOL_PATH,
            expected_source_pairwise_protocol_sha256=fixture[
                "source_protocol_sha256"
            ],
            source_feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_source_feature_contract_sha256=fixture[
                "feature_contract_sha256"
            ],
            historical_registry_descriptor_path=fixture[
                "registry_descriptor_path"
            ],
            expected_historical_registry_descriptor_sha256=fixture[
                "registry_descriptor_sha256"
            ],
            historical_ranker_descriptor_path=fixture["ranker_descriptor_path"],
            expected_historical_ranker_descriptor_sha256=fixture[
                "ranker_descriptor_sha256"
            ],
            historical_ranker_manifest_path=fixture["ranker_manifest_path"],
            expected_historical_ranker_manifest_sha256=fixture[
                "ranker_manifest_sha256"
            ],
            freeze_created_at_ts=3_000,
            active_lineage_state_paths=active_state_paths,
            final_prior_quarantine_path=quarantine_path,
            expected_final_prior_quarantine_sha256=(
                quarantine_sha256
                if quarantine_sha256 is not None
                else (_sha256(quarantine_path) if quarantine_path else None)
            ),
            overwrite_existing=overwrite,
        )
    )


def _assert_blocked_safety(payload: dict[str, Any]) -> None:
    assert payload["paper_only"] is True
    assert payload["capital_at_risk"] is False
    assert payload["polymarket_write_enabled"] is False
    assert payload["wallet_signing_enabled"] is False
    assert payload["source_model_candidate_eligible"] is False
    assert payload["freeze_ready"] is False
    assert payload["promotion_evidence_eligible"] is False
    assert payload["v8_execution_handoff_allowed"] is False
    assert payload["#134_resume_allowed"] is False
    assert payload["#146_start_allowed"] is False


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
