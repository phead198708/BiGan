from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    INDEX_ENTRY_SCHEMA_VERSION,
    ZERO_SHA256,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    CANDIDATE_NAME,
    _blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_future import (
    CANDIDATE_MANIFEST_SCHEMA_VERSION,
    PolicySelectedConformalV6FuturePreRegistrationConfig,
    pre_register_policy_selected_conformal_v6_future_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROFILE = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_policy_selected_conformal_net_return_v6_preregistration_v1.json"
)
COLLECTOR_PROTOCOL = (
    ROOT
    / "examples/v8/polymarket_configs/"
    "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)


def test_v6_future_preregistration_freezes_prefix_and_prior_identities(
    tmp_path: Path,
) -> None:
    fixture = _future_prereg_fixture(tmp_path)
    result = pre_register_policy_selected_conformal_v6_future_evaluation(
        _config(tmp_path, fixture)
    )
    report = result["report"]
    boundary = result["source_boundary"]
    assert report["future_preregistration_ready"] is True
    assert report["collector_index_prefix_entry_count"] == 1
    assert report["minimum_collection_index_sequence"] == 2
    assert report["target_quality_valid_market_count"] == 300
    assert report["minimum_guard_accepted_unique_market_count"] == 120
    assert boundary["minimum_collection_decision_ts"] == 201
    assert boundary["prior_market_ids"] == ["prior-market"]
    assert boundary["prior_slugs"] == ["prior-slug"]
    assert boundary["prior_source_row_hashes"] == ["b" * 64]
    assert boundary["labels_outcomes_or_pnl_opened"] is False
    assert result["manifest"]["future_labels_outcomes_or_pnl_opened"] is False
    assert result["manifest"]["paper_candidate_allowed"] is False
    assert result["manifest"]["v8_execution_handoff_allowed"] is False


def test_v6_future_preregistration_rejects_candidate_outcome_leakage(
    tmp_path: Path,
) -> None:
    fixture = _future_prereg_fixture(tmp_path)
    candidate = _load_json(fixture["candidate_manifest"])
    candidate["uses_204_outcomes_for_fitting"] = True
    _write_json(fixture["candidate_manifest"], candidate)
    fixture["candidate_manifest_sha256"] = _sha256(fixture["candidate_manifest"])
    with pytest.raises(ValueError, match="issue204_outcomes"):
        pre_register_policy_selected_conformal_v6_future_evaluation(
            _config(tmp_path, fixture)
        )


def test_v6_future_preregistration_rejects_mutated_index_pin(tmp_path: Path) -> None:
    fixture = _future_prereg_fixture(tmp_path)
    fixture["collector_index"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pre_register_policy_selected_conformal_v6_future_evaluation(
            _config(tmp_path, fixture)
        )


def _config(
    tmp_path: Path,
    fixture: dict[str, Path | str],
) -> PolicySelectedConformalV6FuturePreRegistrationConfig:
    return PolicySelectedConformalV6FuturePreRegistrationConfig(
        run_id="future-prereg",
        output_dir=tmp_path / "runs",
        candidate_manifest_path=fixture["candidate_manifest"],
        expected_candidate_manifest_sha256=str(fixture["candidate_manifest_sha256"]),
        baseline_manifest_path=fixture["baseline_manifest"],
        expected_baseline_manifest_sha256=str(fixture["baseline_manifest_sha256"]),
        collector_protocol_path=COLLECTOR_PROTOCOL,
        expected_collector_protocol_sha256=_sha256(COLLECTOR_PROTOCOL),
        collector_index_path=fixture["collector_index"],
        expected_collector_index_sha256=str(fixture["collector_index_sha256"]),
        builder_git_commit="1" * 40,
        preregistration_created_ts=200,
    )


def _future_prereg_fixture(tmp_path: Path) -> dict[str, Path | str]:
    model_path = tmp_path / "v6-model.json"
    model_path.write_text("v6-model\n", encoding="utf-8")
    calibration_path = tmp_path / "v6-calibration.json"
    _write_json(calibration_path, {"frozen": True})
    baseline_model_path = tmp_path / "v4-model.json"
    baseline_model_path.write_text("v4-model\n", encoding="utf-8")
    baseline_profile_path = tmp_path / "v4-profile.json"
    _write_json(baseline_profile_path, {"frozen": True})
    profile = _load_json(SOURCE_PROFILE)
    profile["frozen_upstream"]["matched_v4_model_sha256"] = _sha256(baseline_model_path)
    profile_path = tmp_path / "v6-profile.json"
    _write_json(profile_path, profile)
    candidate = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "candidate_freeze_created_ts": 100,
        "research_candidate_frozen": True,
        "calibration_gate_passed": True,
        "candidate_specific_future_evaluation_allowed": True,
        "candidate_specific_future_evaluation_blocking_reason_codes": [],
        "model": _descriptor(model_path),
        "model_sha256": _sha256(model_path),
        "calibration_artifact": _descriptor(calibration_path),
        "profile": _descriptor(profile_path),
        "uses_204_outcomes_for_fitting": False,
        "uses_204_pnl_for_tuning": False,
        "policy_pnl_computed": False,
        "calibration_check_labels_opened_by_fit": False,
        **_blocked_safety_fields(),
    }
    candidate_path = tmp_path / "v6-candidate.json"
    _write_json(candidate_path, candidate)
    baseline = {
        "candidate_name": "guard_compatible_direct_net_return_v4",
        "research_candidate_frozen": True,
        "candidate_specific_future_evaluation_allowed": True,
        "current_oof_validation_or_future_pnl_used_for_tuning": False,
        "model": _descriptor(baseline_model_path),
        "fit_profile": _descriptor(baseline_profile_path),
        **_blocked_safety_fields(),
    }
    baseline_path = tmp_path / "v4-candidate.json"
    _write_json(baseline_path, baseline)
    index_path = tmp_path / "collector-index.jsonl"
    row = {
        "schema_version": INDEX_ENTRY_SCHEMA_VERSION,
        "sequence": 1,
        "previous_entry_sha256": ZERO_SHA256,
        "decision_id": "a" * 64,
        "source_row_hash": "b" * 64,
        "market_id": "prior-market",
        "slug": "prior-slug",
        "market_start_ts": 10,
        "market_end_ts": 50,
        "scheduled_round_start_ts": 10,
        "capture_quality_valid": True,
        "labels_outcomes_or_pnl_opened": False,
        **_blocked_safety_fields(),
    }
    row["entry_sha256"] = canonical_json_sha256(row)
    index_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "candidate_manifest": candidate_path,
        "candidate_manifest_sha256": _sha256(candidate_path),
        "baseline_manifest": baseline_path,
        "baseline_manifest_sha256": _sha256(baseline_path),
        "collector_index": index_path,
        "collector_index_sha256": _sha256(index_path),
    }


def _descriptor(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
