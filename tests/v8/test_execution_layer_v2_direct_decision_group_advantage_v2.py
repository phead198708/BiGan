from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_direct_decision_group_advantage_v2 import (
    DirectDecisionGroupAdvantageV2PreRegistrationConfig,
    freeze_direct_decision_group_advantage_v2_pre_registration,
    validate_direct_decision_group_advantage_v2_protocol,
)

PROTOCOL_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_direct_decision_group_action_advantage_v2.json"
)


def test_issue197_protocol_freezes_direct_estimands_and_honest_uncertainty() -> None:
    protocol = _load_json(PROTOCOL_PATH)

    validate_direct_decision_group_advantage_v2_protocol(protocol)

    assert set(protocol["training_estimands"]) >= {
        "absolute_post_cost_net_return",
        "advantage_vs_no_trade",
        "advantage_vs_best_alternative",
    }
    assert protocol["decision_rule"]["trade_must_pass_all_lower_confidence_bounds"] is True
    assert protocol["calibration_protocol"]["duplicate_quantile_boundaries_must_merge"] is True
    assert (
        protocol["calibration_protocol"]["bootstrap_complete_shrunken_estimator_required"] is True
    )
    assert (
        protocol["calibration_protocol"]["convex_combination_of_separately_estimated_lcbs_allowed"]
        is False
    )
    assert protocol["quarantined_lineage"]["eligible_fit_role"] == ("development_train")
    assert protocol["future_evaluation_protocol"]["fixed_attempt_count_batch_required"] is False
    assert (
        protocol["future_evaluation_protocol"][
            "future_holdout_collection_may_precede_candidate_fit"
        ]
        is True
    )
    assert (
        protocol["future_evaluation_protocol"][
            "future_labels_must_remain_sealed_until_candidate_freeze"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        (
            "calibration_protocol",
            "duplicate_quantile_boundaries_must_merge",
            False,
            "reachable_adaptive_buckets",
        ),
        (
            "calibration_protocol",
            "convex_combination_of_separately_estimated_lcbs_allowed",
            True,
            "full_estimator_bootstrap",
        ),
        (
            "calibration_protocol",
            "current_issue189_oof_files_may_be_opened",
            True,
            "new_internal_calibration_only",
        ),
        (
            "future_evaluation_protocol",
            "result_dependent_extension_allowed",
            True,
            "strict_future_selection",
        ),
    ],
)
def test_issue197_protocol_rejects_semantic_drift(
    section: str,
    key: str,
    value: object,
    reason: str,
) -> None:
    protocol = _load_json(PROTOCOL_PATH)
    protocol[section][key] = value

    with pytest.raises(ValueError, match=reason):
        validate_direct_decision_group_advantage_v2_protocol(protocol)


def test_issue197_pre_registration_is_deterministic_and_opens_no_labels(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)

    first = freeze_direct_decision_group_advantage_v2_pre_registration(config)
    second = freeze_direct_decision_group_advantage_v2_pre_registration(
        replace(config, overwrite_existing=True)
    )

    assert first["report_sha256"] == second["report_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    report = second["report"]
    assert report["pre_registration_ready"] is True
    assert report["fit_eligible_market_count"] == 90
    assert report["quarantined_market_count"] == 105
    assert report["feature_row_files_opened"] is False
    assert report["label_outcome_or_pnl_files_opened"] is False
    assert report["current_issue189_oof_files_opened"] is False
    assert report["current_oof_validation_or_confirmatory_pnl_used"] is False
    assert report["new_label_access_allowed_by_this_issue"] is False
    assert report["minimum_future_decision_ts_exclusive"] == 1_800_000_000_000
    assert report["future_quality_valid_market_target"] == 220
    assert report["future_accepted_unique_market_target"] == 88
    assert report["future_holdout_collection_may_precede_candidate_fit"] is True
    assert report["future_labels_must_remain_sealed_until_candidate_freeze"] is True
    assert report["future_holdout_evaluation_requires_candidate_freeze"] is True
    assert report["fitting_or_prediction_attempted"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["freeze_ready"] is False
    assert report["promotion_evidence_eligible"] is False
    assert report["v8_execution_handoff_allowed"] is False
    assert report["paper_candidate_allowed"] is False
    assert report["paper_only"] is True
    assert report["capital_at_risk"] is False


def test_issue197_pre_registration_rejects_target_fields_before_freeze(
    tmp_path: Path,
) -> None:
    config = _fixture_config(tmp_path)
    role_manifest = _load_json(config.role_assignment_manifest_path)
    selected_path = Path(role_manifest["selected_rows"]["path"])
    rows = _load_jsonl(selected_path)
    rows[0]["settlement_outcome"] = "UP"
    selected_descriptor = _write_jsonl(selected_path, rows)
    role_manifest["selected_rows"] = selected_descriptor
    role_path = config.role_assignment_manifest_path
    _write_json(role_path, role_manifest)
    protocol = _load_json(config.protocol_path)
    protocol["quarantined_lineage"]["selected_rows_sha256"] = selected_descriptor["sha256"]
    protocol["quarantined_lineage"]["role_assignment_manifest_sha256"] = _sha256(role_path)
    _write_json(config.protocol_path, protocol)
    drifted = replace(
        config,
        expected_protocol_sha256=_sha256(config.protocol_path),
        expected_role_assignment_manifest_sha256=_sha256(role_path),
    )

    with pytest.raises(ValueError, match="forbidden target fields"):
        freeze_direct_decision_group_advantage_v2_pre_registration(drifted)


def _fixture_config(
    tmp_path: Path,
) -> DirectDecisionGroupAdvantageV2PreRegistrationConfig:
    safety = {
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
    feature_path = tmp_path / "feature_contract.json"
    _write_json(feature_path, {"schema_version": "test-feature-contract"})
    selected_path = tmp_path / "role_rows.jsonl"
    roles = (
        ["development_train"] * 90
        + ["development_calibration"] * 45
        + ["confirmatory_validation"] * 60
    )
    selected_descriptor = _write_jsonl(
        selected_path,
        [
            {
                "market_id": f"market-{index:03d}",
                "role": role,
                "selection_rank": index,
            }
            for index, role in enumerate(roles, start=1)
        ],
    )
    role_path = tmp_path / "role_manifest.json"
    _write_json(
        role_path,
        {
            "role_assignment_ready": True,
            "labels_or_outcomes_opened_for_role_assignment": False,
            "selected_rows": selected_descriptor,
            "feature_contract": _descriptor(feature_path),
            **safety,
        },
    )
    power_design_path = tmp_path / "power_design.json"
    _write_json(
        power_design_path,
        {
            "uses_current_oof_validation_or_confirmatory_pnl": False,
            "uses_realized_candidate_pnl_for_design": False,
        },
    )
    power_report_path = tmp_path / "power_report.json"
    _write_json(
        power_report_path,
        {
            "power_analysis_ready": True,
            "uses_current_oof_validation_or_confirmatory_pnl": False,
            "uses_realized_candidate_pnl_for_design": False,
            "recommended_quality_valid_market_count": 220,
            "recommended_required_accepted_unique_market_count": 88,
            "result_dependent_extension_allowed": False,
        },
    )
    issue190_path = tmp_path / "issue190_freeze.json"
    _write_json(
        issue190_path,
        {
            "collection_control_is_outcome_blind": True,
            "labels_or_outcomes_opened_for_collection_freeze": False,
            "settlement_finalizer_started_during_collection": False,
            "training_corpus_export_during_collection_allowed": False,
            "target_valid_market_count": 220,
            **safety,
        },
    )
    persistent_path = tmp_path / "persistent_protocol.json"
    _write_json(
        persistent_path,
        {
            "frozen": True,
            "outcome_blind_collection_only": True,
            "settlement_finalizer_enabled": False,
            "resolution_provider_enabled": False,
            "training_corpus_export_enabled": False,
            "labels_outcomes_or_pnl_opened": False,
            "append_only_index": {"hash_chain_required": True},
            **safety,
        },
    )
    protocol = _load_json(PROTOCOL_PATH)
    protocol["quarantined_lineage"]["role_assignment_manifest_sha256"] = _sha256(role_path)
    protocol["quarantined_lineage"]["selected_rows_sha256"] = selected_descriptor["sha256"]
    protocol["frozen_lineage_hashes"] = {
        "feature_contract_sha256": _sha256(feature_path),
        "power_design_sha256": _sha256(power_design_path),
        "power_report_sha256": _sha256(power_report_path),
        "issue190_collection_freeze_sha256": _sha256(issue190_path),
        "issue192_persistent_collector_protocol_sha256": _sha256(persistent_path),
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)
    return DirectDecisionGroupAdvantageV2PreRegistrationConfig(
        run_id="issue197-test",
        output_dir=tmp_path / "runs",
        freeze_created_at_ts=1_800_000_000_000,
        protocol_path=protocol_path,
        expected_protocol_sha256=_sha256(protocol_path),
        role_assignment_manifest_path=role_path,
        expected_role_assignment_manifest_sha256=_sha256(role_path),
        power_design_path=power_design_path,
        expected_power_design_sha256=_sha256(power_design_path),
        power_report_path=power_report_path,
        expected_power_report_sha256=_sha256(power_report_path),
        issue190_collection_freeze_path=issue190_path,
        expected_issue190_collection_freeze_sha256=_sha256(issue190_path),
        persistent_collector_protocol_path=persistent_path,
        expected_persistent_collector_protocol_sha256=_sha256(persistent_path),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return _descriptor(path)


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
