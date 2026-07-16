"""Tests for frozen-ranker fresh calibration and confirmatory separation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pnl_aligned_action_value import (
    REQUIRED_ACTIONS,
)
from bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration import (
    CALIBRATION_MARKET_COUNT,
    CALIBRATION_ROLE,
    CONFIRMATORY_ROLE,
    HybridPairwiseCalibrationReadinessConfig,
    HybridPairwiseConfirmatoryConfig,
    HybridPairwiseFreshCalibrationConfig,
    _score_only_oof_rows,
    _validate_fresh_role_lineage,
    _validate_role_assignment,
    evaluate_hybrid_pairwise_calibration_readiness,
    evaluate_hybrid_pairwise_confirmatory_once,
    freeze_hybrid_pairwise_fresh_calibration,
)

ROOT = Path(__file__).resolve().parents[2]
HYBRID_PROTOCOL_PATH = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_hybrid_pairwise_fresh_calibration_v1.json"
)
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


def test_historical_oof_bucket_input_ignores_target_values(
    tmp_path: Path,
) -> None:
    rows = _oof_rows()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_jsonl(first_path, rows)
    mutated = [
        {
            **row,
            "target_net_pnl_per_contract": (
                100_000.0 + index if index % 2 else -100_000.0 - index
            ),
        }
        for index, row in enumerate(rows)
    ]
    _write_jsonl(second_path, mutated)

    first, first_audit = _score_only_oof_rows(first_path)
    second, second_audit = _score_only_oof_rows(second_path)

    assert first == second
    assert all("target_net_pnl_per_contract" not in row for row in first)
    assert first_audit["target_field_present_count"] == len(rows)
    assert second_audit["target_field_present_count"] == len(rows)
    assert first_audit["target_field_values_used_for_bucket_construction"] is False
    assert second_audit["target_field_values_used_for_bucket_construction"] is False


def test_readiness_writes_blocked_artifact_before_issue183_and_roles_complete(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    result = _run_readiness(tmp_path, fixture, include_roles=False)
    report = result["report"]

    assert report["readiness_status"] == "blocked_fail_closed"
    assert report["calibration_readiness_passed"] is False
    assert report["calibration_start_allowed"] is False
    assert report["model_prediction_attempted"] is False
    assert report["ranker_retraining_attempted"] is False
    assert report["label_or_outcome_artifacts_opened"] is False
    assert report["blocking_reason_codes"] == [
        "fresh_45_60_role_assignment_missing",
        "issue183_terminal_freeze_incomplete",
    ]
    _assert_blocked_safety(report)


def test_readiness_passes_only_after_terminal_freeze_and_valid_45_60_roles(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state = _load_json(fixture["upstream_state_path"])
    state.update(
        {
            "status": "completed",
            "precollection_readiness_passed": True,
            "precollection_freeze_created": True,
            "collection_start_allowed": False,
            "collection_start_command_generated": False,
        }
    )
    _write_json(fixture["upstream_state_path"], state)
    fixture["upstream_state_sha256"] = _sha256(
        fixture["upstream_state_path"]
    )

    result = _run_readiness(tmp_path, fixture, include_roles=True)
    report = result["report"]

    assert report["readiness_status"] == "ready_for_fresh_calibration"
    assert report["calibration_readiness_passed"] is True
    assert report["calibration_start_allowed"] is True
    assert report["fresh_role_market_counts"] == {
        CALIBRATION_ROLE: 45,
        CONFIRMATORY_ROLE: 60,
    }
    assert report["model_prediction_attempted"] is False
    assert report["label_or_outcome_artifacts_opened"] is False
    _assert_blocked_safety(report)


def test_oof_score_only_requires_complete_five_action_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete.jsonl"
    _write_jsonl(path, _oof_rows()[:-1])

    with pytest.raises(
        ValueError,
        match="complete five-action decision grid",
    ):
        _score_only_oof_rows(path)


def test_role_assignment_requires_exact_45_60_chronological_split(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = _load_json(fixture["role_manifest_path"])
    rows = _jsonl(fixture["role_rows_path"])

    _validate_role_assignment(manifest, rows)

    rows[-1]["role"] = CALIBRATION_ROLE
    with pytest.raises(ValueError, match="counts do not match 45/60"):
        _validate_role_assignment(manifest, rows)


def test_role_lineage_rejects_final_quarantine_overlap(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    role_manifest = _load_json(fixture["role_manifest_path"])
    role_rows = _jsonl(fixture["role_rows_path"])
    quarantine_path = fixture["quarantine_path"]
    quarantine = _load_json(quarantine_path)
    quarantine["prior_market_ids"] = [role_rows[0]["market_id"]]
    _write_json(quarantine_path, quarantine)
    quarantine_descriptor = _descriptor(quarantine_path)
    role_manifest["final_prior_lineage_quarantine"] = quarantine_descriptor

    with pytest.raises(
        ValueError,
        match="overlaps final quarantine",
    ):
        _validate_fresh_role_lineage(
            role_manifest=role_manifest,
            role_rows=role_rows,
            hybrid_protocol=_load_json(fixture["hybrid_protocol_path"]),
            precollection_freeze_descriptor=_descriptor(
                fixture["precollection_freeze_path"]
            ),
            final_quarantine_descriptor=quarantine_descriptor,
        )


def test_calibration_uses_frozen_model_and_never_opens_confirmatory_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls = _patch_pipeline(
        monkeypatch,
        development_gate_passed=True,
        confirmatory_gate_passed=True,
    )
    monkeypatch.setattr(
        "xgboost.train",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ranker retraining is forbidden")
        ),
    )

    result = _run_calibration(tmp_path, fixture)
    freeze = result["freeze_manifest"]

    assert calls["materialized_roles"] == [CALIBRATION_ROLE]
    assert freeze["development_gate_passed"] is True
    assert freeze["calibration_frozen"] is True
    assert freeze["confirmatory_evaluation_started"] is False
    assert freeze["confirmatory_labels_opened"] is False
    assert freeze["confirmatory_label_access_allowed"] is True
    assert freeze["ranker_retrained"] is False
    assert freeze["ranker_score_mutated"] is False
    assert freeze["uses_current_oof_or_validation_pnl_for_tuning"] is False
    _assert_blocked_safety(freeze)
    identity = _load_json(result["identity_report_path"])
    assert identity["model_sha256_unchanged_after_prediction"] is True
    assert identity["ranker_retrained"] is False
    leakage = _load_json(
        Path(freeze["calibration_leakage_role_audit"]["path"])
    )
    assert leakage["leakage_and_role_audit_passed"] is True
    assert leakage["confirmatory_labels_opened"] is False


def test_development_gate_failure_blocks_confirmatory_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls = _patch_pipeline(
        monkeypatch,
        development_gate_passed=False,
        confirmatory_gate_passed=True,
    )
    calibration = _run_calibration(tmp_path, fixture)

    with pytest.raises(ValueError, match="development gate did not pass"):
        evaluate_hybrid_pairwise_confirmatory_once(
            HybridPairwiseConfirmatoryConfig(
                run_id="confirmatory",
                output_dir=tmp_path / "runs",
                calibration_freeze_manifest_path=calibration[
                    "freeze_manifest_path"
                ],
                expected_calibration_freeze_manifest_sha256=calibration[
                    "freeze_manifest_sha256"
                ],
            )
        )

    assert calls["materialized_roles"] == [CALIBRATION_ROLE]
    assert not (tmp_path / "runs" / ".hybrid_pairwise_confirmatory_claims").exists()


def test_confirmatory_is_one_shot_and_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls = _patch_pipeline(
        monkeypatch,
        development_gate_passed=True,
        confirmatory_gate_passed=True,
    )
    calibration = _run_calibration(tmp_path, fixture)
    config = HybridPairwiseConfirmatoryConfig(
        run_id="confirmatory-first",
        output_dir=tmp_path / "runs",
        calibration_freeze_manifest_path=calibration[
            "freeze_manifest_path"
        ],
        expected_calibration_freeze_manifest_sha256=calibration[
            "freeze_manifest_sha256"
        ],
    )

    first = evaluate_hybrid_pairwise_confirmatory_once(config)
    candidate = first["candidate_freeze"]

    assert calls["materialized_roles"] == [
        CALIBRATION_ROLE,
        CONFIRMATORY_ROLE,
    ]
    assert candidate["confirmatory_gate_passed"] is True
    assert candidate["diagnostic_candidate_evidence_passed"] is True
    assert candidate["source_model_candidate_eligible"] is False
    assert candidate["promotion_evidence_eligible"] is False
    assert candidate["future_unseen_execution_holdout_required"] is True
    _assert_blocked_safety(candidate)
    claim = _load_json(first["claim_path"])
    assert claim["confirmatory_labels_opened"] is True
    assert claim["evaluation_completed"] is True

    with pytest.raises(
        ValueError,
        match="confirmatory evaluation was already claimed",
    ):
        evaluate_hybrid_pairwise_confirmatory_once(
            HybridPairwiseConfirmatoryConfig(
                run_id="confirmatory-second",
                output_dir=tmp_path / "runs",
                calibration_freeze_manifest_path=calibration[
                    "freeze_manifest_path"
                ],
                expected_calibration_freeze_manifest_sha256=calibration[
                    "freeze_manifest_sha256"
                ],
            )
        )
    assert calls["materialized_roles"] == [
        CALIBRATION_ROLE,
        CONFIRMATORY_ROLE,
    ]


def test_ranker_identity_drift_fails_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls = _patch_pipeline(
        monkeypatch,
        development_gate_passed=True,
        confirmatory_gate_passed=True,
    )
    descriptor = _load_json(fixture["ranker_descriptor_path"])
    descriptor["model_sha256"] = "f" * 64
    _write_json(fixture["ranker_descriptor_path"], descriptor)
    fixture["ranker_descriptor_sha256"] = _sha256(
        fixture["ranker_descriptor_path"]
    )
    hybrid = _load_json(fixture["hybrid_protocol_path"])
    hybrid["historical_ranker_freeze"]["descriptor_sha256"] = fixture[
        "ranker_descriptor_sha256"
    ]
    _write_json(fixture["hybrid_protocol_path"], hybrid)
    fixture["hybrid_protocol_sha256"] = _sha256(
        fixture["hybrid_protocol_path"]
    )

    with pytest.raises(
        ValueError,
        match="frozen historical ranker identity mismatch",
    ):
        _run_calibration(tmp_path, fixture)
    assert calls["materialized_roles"] == []


def test_confirmatory_failure_keeps_claim_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    calls = _patch_pipeline(
        monkeypatch,
        development_gate_passed=True,
        confirmatory_gate_passed=False,
    )
    calibration = _run_calibration(tmp_path, fixture)
    result = evaluate_hybrid_pairwise_confirmatory_once(
        HybridPairwiseConfirmatoryConfig(
            run_id="confirmatory-failed-gate",
            output_dir=tmp_path / "runs",
            calibration_freeze_manifest_path=calibration[
                "freeze_manifest_path"
            ],
            expected_calibration_freeze_manifest_sha256=calibration[
                "freeze_manifest_sha256"
            ],
        )
    )

    assert calls["materialized_roles"][-1] == CONFIRMATORY_ROLE
    assert result["confirmatory_gate_passed"] is False
    assert result["candidate_freeze"]["diagnostic_candidate_evidence_passed"] is False
    assert result["candidate_freeze"]["source_model_candidate_eligible"] is False
    assert result["candidate_freeze"]["promotion_evidence_eligible"] is False
    _assert_blocked_safety(result["candidate_freeze"])


def test_confirmatory_claim_records_label_access_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_pipeline(
        monkeypatch,
        development_gate_passed=True,
        confirmatory_gate_passed=True,
    )
    calibration = _run_calibration(tmp_path, fixture)

    def fail_materialization(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("fixture confirmatory materialization failure")

    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_materialize_fresh_action_rows",
        fail_materialization,
    )
    with pytest.raises(
        ValueError,
        match="fixture confirmatory materialization failure",
    ):
        evaluate_hybrid_pairwise_confirmatory_once(
            HybridPairwiseConfirmatoryConfig(
                run_id="confirmatory-materialization-failure",
                output_dir=tmp_path / "runs",
                calibration_freeze_manifest_path=calibration[
                    "freeze_manifest_path"
                ],
                expected_calibration_freeze_manifest_sha256=calibration[
                    "freeze_manifest_sha256"
                ],
            )
        )

    claims = list(
        (tmp_path / "runs" / ".hybrid_pairwise_confirmatory_claims").glob(
            "*.json"
        )
    )
    assert len(claims) == 1
    claim = _load_json(claims[0])
    assert claim["confirmatory_labels_opened"] is True
    assert claim["evaluation_completed"] is False
    assert claim["evaluation_failed_closed"] is True


def _fixture(tmp_path: Path) -> dict[str, Any]:
    source_protocol_sha = _sha256(SOURCE_PROTOCOL_PATH)
    feature_contract_sha = _sha256(FEATURE_CONTRACT_PATH)
    model_path = tmp_path / "frozen-model.json"
    model_path.write_text('{"frozen":true}\n', encoding="utf-8")
    model_sha = _sha256(model_path)
    oof_path = tmp_path / "historical-oof.jsonl"
    _write_jsonl(oof_path, _oof_rows())
    ranker_manifest = {
        "schema_version": "fixture-ranker-manifest-v1",
        "freeze_id": "1" * 64,
        "model_sha256": model_sha,
        "dataset_hash": "2" * 64,
        "oof_dataset_hash": "3" * 64,
        "split_hash": "4" * 64,
        "model_config_hash": "5" * 64,
        "rank_scores_execution_eligible": False,
        "train_oof_predictions": _descriptor(oof_path),
    }
    ranker_manifest_path = tmp_path / "ranker-manifest.json"
    _write_json(ranker_manifest_path, ranker_manifest)
    ranker_descriptor = {
        "schema_version": "fixture-ranker-descriptor-v1",
        "freeze_id": ranker_manifest["freeze_id"],
        "model_sha256": model_sha,
        "dataset_hash": ranker_manifest["dataset_hash"],
        "split_hash": ranker_manifest["split_hash"],
        "model_config_hash": ranker_manifest["model_config_hash"],
        "model": _descriptor(model_path),
        "train_oof_predictions": _descriptor(oof_path),
        "rank_scores_execution_eligible": False,
    }
    ranker_descriptor_path = tmp_path / "ranker-descriptor.json"
    _write_json(ranker_descriptor_path, ranker_descriptor)

    hybrid_protocol = deepcopy(_load_json(HYBRID_PROTOCOL_PATH))
    hybrid_protocol["source_pairwise_protocol_sha256"] = source_protocol_sha
    hybrid_protocol["source_feature_contract_sha256"] = feature_contract_sha
    hybrid_protocol["historical_ranker_freeze"].update(
        {
            "descriptor_sha256": _sha256(ranker_descriptor_path),
            "freeze_id": ranker_manifest["freeze_id"],
            "model_sha256": model_sha,
            "dataset_hash": ranker_manifest["dataset_hash"],
            "oof_dataset_hash": ranker_manifest["oof_dataset_hash"],
            "split_hash": ranker_manifest["split_hash"],
            "model_config_hash": ranker_manifest["model_config_hash"],
        }
    )
    hybrid_protocol_path = tmp_path / "hybrid-protocol.json"
    _write_json(hybrid_protocol_path, hybrid_protocol)

    quarantine = {
        "schema_version": "fixture-final-quarantine-v1",
        "status": "prior_lineage_complete",
        "final": True,
        "active_prior_lineage_complete": True,
        "includes_issue175_through_issue179": True,
        "maximum_prior_decision_ts": 8_000,
        "minimum_future_decision_ts": 8_001,
        "prior_market_ids": ["prior-a", "prior-b"],
        **_blocked_safety_fields(),
    }
    quarantine_path = tmp_path / "final-quarantine.json"
    _write_json(quarantine_path, quarantine)
    precollection_freeze = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-precollection-freeze-manifest-v1"
        ),
        "candidate_lineage": hybrid_protocol["candidate_lineage"],
        "minimum_collection_decision_ts": 9_000,
        "fresh_role_plan": hybrid_protocol["fresh_role_plan"],
        "collection_plan": hybrid_protocol["collection_plan"],
        "ranker_retraining_allowed": False,
        "ranker_score_mutation_allowed": False,
        **_blocked_safety_fields(),
    }
    precollection_freeze_path = tmp_path / "precollection-freeze.json"
    _write_json(precollection_freeze_path, precollection_freeze)
    precollection_descriptor = _descriptor(precollection_freeze_path)
    quarantine_descriptor = _descriptor(quarantine_path)

    role_rows = []
    for index in range(105):
        calibration = index < CALIBRATION_MARKET_COUNT
        role = CALIBRATION_ROLE if calibration else CONFIRMATORY_ROLE
        timestamp = 10_000 + index if calibration else 20_000 + index
        role_rows.append(
            {
                "selection_rank": index + 1,
                "role": role,
                "market_id": f"fresh-{index:03d}",
                "minimum_decision_ts": timestamp,
                "maximum_decision_ts": timestamp,
                "source_corpus_dir": str(tmp_path / f"corpus-{index:03d}"),
                "corpus_manifest": {
                    "path": str(tmp_path / f"manifest-{index:03d}.json"),
                    "sha256": f"{index % 10}" * 64,
                },
                "execution_compatibility_validated_before_label_access": True,
                "labels_or_outcomes_opened_for_role_assignment": False,
                "source_precollection_freeze_sha256": (
                    precollection_descriptor["sha256"]
                ),
                "source_final_quarantine_sha256": (
                    quarantine_descriptor["sha256"]
                ),
            }
        )
    role_rows_path = tmp_path / "role-rows.jsonl"
    _write_jsonl(role_rows_path, role_rows)
    role_manifest = {
        "schema_version": (
            "bigan-v8-hybrid-pairwise-fresh-role-assignment-v1"
        ),
        "role_assignment_ready": True,
        "selected_market_count": 105,
        "selected_rows": _descriptor(role_rows_path),
        "hybrid_precollection_freeze": precollection_descriptor,
        "final_prior_lineage_quarantine": quarantine_descriptor,
        "labels_or_outcomes_opened_for_role_assignment": False,
        "prior_market_overlap_count": 0,
        "role_market_overlap_count": 0,
        "chronology_validation_passed": True,
        **_blocked_safety_fields(),
    }
    role_manifest_path = tmp_path / "role-manifest.json"
    _write_json(role_manifest_path, role_manifest)
    upstream_state = {
        "status": "waiting_for_issue179_terminal_state",
        "precollection_readiness_passed": False,
        "precollection_freeze_created": False,
        "collection_start_allowed": False,
        "collection_start_command_generated": False,
        "paper_only": True,
        "capital_at_risk": False,
    }
    upstream_state_path = tmp_path / "upstream-state.json"
    _write_json(upstream_state_path, upstream_state)
    return {
        "hybrid_protocol_path": hybrid_protocol_path,
        "hybrid_protocol_sha256": _sha256(hybrid_protocol_path),
        "ranker_descriptor_path": ranker_descriptor_path,
        "ranker_descriptor_sha256": _sha256(ranker_descriptor_path),
        "ranker_manifest_path": ranker_manifest_path,
        "ranker_manifest_sha256": _sha256(ranker_manifest_path),
        "role_manifest_path": role_manifest_path,
        "role_manifest_sha256": _sha256(role_manifest_path),
        "role_rows_path": role_rows_path,
        "quarantine_path": quarantine_path,
        "precollection_freeze_path": precollection_freeze_path,
        "upstream_state_path": upstream_state_path,
        "upstream_state_sha256": _sha256(upstream_state_path),
    }


def _run_readiness(
    tmp_path: Path,
    fixture: dict[str, Any],
    *,
    include_roles: bool,
) -> dict[str, Any]:
    return evaluate_hybrid_pairwise_calibration_readiness(
        HybridPairwiseCalibrationReadinessConfig(
            run_id=(
                "readiness-with-roles"
                if include_roles
                else "readiness-blocked"
            ),
            output_dir=tmp_path / "readiness-runs",
            hybrid_protocol_path=fixture["hybrid_protocol_path"],
            expected_hybrid_protocol_sha256=fixture[
                "hybrid_protocol_sha256"
            ],
            historical_ranker_descriptor_path=fixture[
                "ranker_descriptor_path"
            ],
            expected_historical_ranker_descriptor_sha256=fixture[
                "ranker_descriptor_sha256"
            ],
            historical_ranker_manifest_path=fixture[
                "ranker_manifest_path"
            ],
            expected_historical_ranker_manifest_sha256=fixture[
                "ranker_manifest_sha256"
            ],
            upstream_terminal_freeze_state_path=fixture[
                "upstream_state_path"
            ],
            expected_upstream_terminal_freeze_state_sha256=fixture[
                "upstream_state_sha256"
            ],
            fresh_role_assignment_manifest_path=(
                fixture["role_manifest_path"] if include_roles else None
            ),
            expected_fresh_role_assignment_manifest_sha256=(
                fixture["role_manifest_sha256"] if include_roles else None
            ),
        )
    )


def _run_calibration(
    tmp_path: Path,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    return freeze_hybrid_pairwise_fresh_calibration(
        HybridPairwiseFreshCalibrationConfig(
            run_id="calibration",
            output_dir=tmp_path / "runs",
            hybrid_protocol_path=fixture["hybrid_protocol_path"],
            expected_hybrid_protocol_sha256=fixture[
                "hybrid_protocol_sha256"
            ],
            source_pairwise_protocol_path=SOURCE_PROTOCOL_PATH,
            expected_source_pairwise_protocol_sha256=_sha256(
                SOURCE_PROTOCOL_PATH
            ),
            feature_contract_path=FEATURE_CONTRACT_PATH,
            expected_feature_contract_sha256=_sha256(
                FEATURE_CONTRACT_PATH
            ),
            historical_ranker_descriptor_path=fixture[
                "ranker_descriptor_path"
            ],
            expected_historical_ranker_descriptor_sha256=fixture[
                "ranker_descriptor_sha256"
            ],
            historical_ranker_manifest_path=fixture[
                "ranker_manifest_path"
            ],
            expected_historical_ranker_manifest_sha256=fixture[
                "ranker_manifest_sha256"
            ],
            fresh_role_assignment_manifest_path=fixture[
                "role_manifest_path"
            ],
            expected_fresh_role_assignment_manifest_sha256=fixture[
                "role_manifest_sha256"
            ],
        )
    )


def _patch_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    development_gate_passed: bool,
    confirmatory_gate_passed: bool,
) -> dict[str, Any]:
    calls: dict[str, Any] = {"materialized_roles": []}

    def fake_materialize(
        role_rows: list[dict[str, Any]],
        *,
        feature_columns: tuple[str, ...],
        expected_market_count: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        del feature_columns
        role = str(role_rows[0]["role"])
        calls["materialized_roles"].append(role)
        assert len(role_rows) == expected_market_count
        rows = _action_rows(role_rows)
        audits = [
            {
                "feature_causality_violation_count": 0,
                "blocking_reason_codes": [],
            }
            for _ in role_rows
        ]
        return rows, audits

    def fake_predict(
        rows: list[dict[str, Any]],
        *,
        booster: Any,
        feature_columns: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        del booster, feature_columns
        score_by_action = {
            action: index / 4.0
            for index, action in enumerate(REQUIRED_ACTIONS)
        }
        return [
            {
                **row,
                "raw_pairwise_rank_score": score_by_action[row["action"]],
                "pairwise_group_normalized_rank_score": score_by_action[
                    row["action"]
                ],
                "ranking_score_source": "model_predicted_pairwise_rank_score",
            }
            for row in rows
        ]

    def fake_artifact(
        calibration_predictions: list[dict[str, Any]],
        *,
        train_oof_predictions: list[dict[str, Any]],
        protocol: dict[str, Any],
        feature_contract_sha256: str,
    ) -> dict[str, Any]:
        del calibration_predictions, train_oof_predictions, protocol
        actions = {
            action: {
                "train_oof_group_normalized_score_tertile_boundaries": [
                    0.33,
                    0.66,
                ]
            }
            for action in REQUIRED_ACTIONS
        }
        groups = {
            f"{action}|{bucket}": {
                "calibrated_action_expected_net_return": (
                    0.0 if action == "NO_TRADE" else 0.05
                ),
                "action_return_lower_confidence_bound": (
                    0.0 if action == "NO_TRADE" else 0.03
                ),
                "estimate_source": "fixture",
            }
            for action in REQUIRED_ACTIONS
            for bucket in ("low", "middle", "high")
        }
        return {
            "schema_version": "fixture-lcb-v1",
            "actions": actions,
            "calibration_groups": groups,
            "feature_contract_sha256": feature_contract_sha256,
            **_blocked_safety_fields(),
        }

    def fake_apply(
        predictions: list[dict[str, Any]],
        *,
        lcb_artifact: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del lcb_artifact
        return [
            {
                **row,
                "calibrated_action_expected_net_return": (
                    0.0 if row["action"] == "NO_TRADE" else 0.05
                ),
                "action_advantage_lcb_net_return": (
                    0.0 if row["action"] == "NO_TRADE" else 0.03
                ),
            }
            for row in predictions
        ]

    def fake_replay(
        predictions: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        del kwargs
        market_ids = sorted({str(row["market_id"]) for row in predictions})
        return [
            {
                "market_id": market_id,
                "execution_guard_order_allowed": True,
                "required_runtime_fields_present": True,
                "settlement_resolved_for_report_only": True,
                "side": "UP" if index % 2 == 0 else "DOWN",
                "action_family": (
                    "HOLD_TO_SETTLEMENT"
                    if index % 2 == 0
                    else "SELL_BEFORE_CLOSE"
                ),
                "paper_only": True,
                "capital_at_risk": False,
            }
            for index, market_id in enumerate(market_ids)
        ]

    metrics = {
        "accepted_bet_count": 40,
        "accepted_unique_market_count": 40,
        "accepted_bet_count_by_side": {"UP": 20, "DOWN": 20},
        "accepted_bet_count_by_family": {
            "HOLD_TO_SETTLEMENT": 20,
            "SELL_BEFORE_CLOSE": 20,
        },
        "net_pnl_sum": 1.0,
        "roi": 0.1,
    }
    robustness = {
        "market_bootstrap_interval_95": {
            "reported": True,
            "lower": 0.1,
        },
        "leave_one_market_out": {
            "reported": True,
            "all_scenarios_positive": True,
        },
        "largest_winner_removal": {
            "reported": True,
            "candidate_net_pnl_after_removal": 0.5,
        },
    }
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_materialize_fresh_action_rows",
        fake_materialize,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_load_frozen_booster",
        lambda path: object(),
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_predict_role_rows",
        fake_predict,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_action_advantage_lcb_artifact",
        fake_artifact,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_apply_action_advantage_lcb_scores",
        fake_apply,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_run_policy_replay",
        fake_replay,
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_accepted_bet_metrics",
        lambda rows: dict(metrics),
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_market_robustness",
        lambda candidate, baseline: dict(robustness),
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_development_freeze_gate",
        lambda **kwargs: {
            "passed": development_gate_passed,
            "checks": {"fixture": development_gate_passed},
            "reason_codes": (
                []
                if development_gate_passed
                else ["fixture_development_gate_failed"]
            ),
        },
    )
    monkeypatch.setattr(
        "bigan.v8.polymarket.training.hybrid_pairwise_frozen_ranker_calibration."
        "_confirmatory_gate",
        lambda **kwargs: {
            "passed": confirmatory_gate_passed,
            "checks": {"fixture": confirmatory_gate_passed},
            "reason_codes": (
                []
                if confirmatory_gate_passed
                else ["fixture_confirmatory_gate_failed"]
            ),
        },
    )
    return calls


def _action_rows(role_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for role_row in role_rows:
        decision_ts = int(role_row["minimum_decision_ts"])
        for action in REQUIRED_ACTIONS:
            rows.append(
                {
                    "market_id": role_row["market_id"],
                    "decision_ts": decision_ts,
                    "max_input_ts": decision_ts,
                    "action": action,
                    "target_net_pnl_per_contract": (
                        0.0 if action == "NO_TRADE" else 0.01
                    ),
                    "target_used_as_decision_input": False,
                    "outcome_fields_used_as_decision_input": False,
                    "paper_only": True,
                    "capital_at_risk": False,
                }
            )
    return rows


def _oof_rows() -> list[dict[str, Any]]:
    rows = []
    for market_index in range(75):
        decision_ts = 1_000 + market_index
        for action_index, action in enumerate(REQUIRED_ACTIONS):
            side = (
                "UP"
                if "BUY_UP" in action
                else "DOWN"
                if "BUY_DOWN" in action
                else "NONE"
            )
            family = (
                "HOLD_TO_SETTLEMENT"
                if "HOLD_TO_SETTLEMENT" in action
                else "SELL_BEFORE_CLOSE"
                if "SELL_BEFORE_CLOSE" in action
                else "NO_TRADE"
            )
            score = action_index / 4.0
            rows.append(
                {
                    "market_id": f"oof-{market_index:03d}",
                    "decision_ts": decision_ts,
                    "action": action,
                    "action_family": family,
                    "side": side,
                    "fold_index": market_index // 15 + 1,
                    "oof_raw_prediction": score,
                    "pairwise_action_rank": 5 - action_index,
                    "pairwise_rank_percentile": score,
                    "pairwise_group_normalized_rank_score": score,
                    "pairwise_group_score_range": 1.0,
                    "pairwise_normalized_margin_vs_no_trade": score,
                    "pairwise_normalized_margin_vs_best_alternative": (
                        score - 0.25
                    ),
                    "pairwise_rank_normalization_scope": (
                        "market_id_decision_ts_five_action_group"
                    ),
                    "raw_rank_score_cross_model_comparison_allowed": False,
                    "target_net_pnl_per_contract": (
                        action_index - 2
                    )
                    / 100.0,
                }
            )
    return rows


def _assert_blocked_safety(payload: dict[str, Any]) -> None:
    for key, value in _blocked_safety_fields().items():
        assert payload[key] is value


def _blocked_safety_fields() -> dict[str, Any]:
    return {
        "paper_only": True,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "v8_execution_handoff_allowed": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
