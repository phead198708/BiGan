from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.exact_model_runtime_binding import (
    ExactModelRuntimeBindingConfig,
    ExactModelRuntimeBindingError,
    verify_exact_model_runtime_binding,
)
from examples.v8.run_execution_layer_v2_persistent_outcome_blind_collector import (
    run_service,
)


def _write(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_fixture(
    tmp_path: Path,
    *,
    booster_binding_drift: bool = False,
    profile_binding_drift: bool = False,
    controller_binding_drift: bool = False,
) -> ExactModelRuntimeBindingConfig:
    candidate_name = "candidate-from-contract"
    booster = xgb.train(
        {"objective": "reg:squarederror", "nthread": 1},
        xgb.DMatrix([[0.0], [1.0]], label=[0.0, 1.0]),
        num_boost_round=1,
    )
    booster_bytes = bytes(booster.save_raw(raw_format="json"))
    booster_base64 = base64.b64encode(booster_bytes).decode("ascii")
    booster_contract_sha256 = canonical_json_sha256(booster_base64)
    profile_path = tmp_path / "profile.json"
    profile_sha256 = _write(
        profile_path,
        {"candidate_name": candidate_name, "frozen": True},
    )
    model_state = {
        "rank_state_id": "e" * 64,
        "rank_lineage_hash": "2" * 64,
        "eligible_prediction_scores_hash": "3" * 64,
        "controller_guard_acceptance_history_hash": "4" * 64,
        "controller_target_outcome_label_or_pnl_free": True,
        "rank_state_uses_target_outcome_or_pnl": False,
        "next_controller_decision": {
            "controller_source": (
                "strictly_prior_full_guard_acceptance_only"
            ),
            "current_market_guard_result_used": False,
            "target_outcome_label_or_pnl_used": False,
        },
    }
    model_path = tmp_path / "model.json"
    model_sha256 = _write(
        model_path,
        {
            "candidate_name": candidate_name,
            "final_weighted_model": {
                "booster_json_base64": booster_base64,
                "booster_sha256": booster_contract_sha256,
            },
            "final_rank_state": model_state,
        },
    )
    binding_state = {
        "rank_state_id": model_state["rank_state_id"],
        "rank_lineage_hash": (
            "0" * 64
            if controller_binding_drift
            else model_state["rank_lineage_hash"]
        ),
        "eligible_prediction_scores_hash": model_state[
            "eligible_prediction_scores_hash"
        ],
        "controller_guard_acceptance_history_hash": model_state[
            "controller_guard_acceptance_history_hash"
        ],
        "controller_state_uses_target_outcome_label_or_pnl": False,
        "future_controller_updates_use_strictly_prior_guard_results_only": (
            True
        ),
    }
    binding_path = tmp_path / "binding.json"
    binding_sha256 = _write(
        binding_path,
        {
            "candidate_name": candidate_name,
            "frozen_model_artifact_sha256": model_sha256,
            "frozen_profile_sha256": (
                "0" * 64 if profile_binding_drift else profile_sha256
            ),
            "frozen_booster_sha256": (
                "0" * 64
                if booster_binding_drift
                else booster_contract_sha256
            ),
            "initial_controller_state": binding_state,
        },
    )
    candidate_contract_path = tmp_path / "candidate_contract.json"
    candidate_contract_sha256 = _write(
        candidate_contract_path,
        {
            "primary_policy": candidate_name,
            "frozen_model_binding_sha256": binding_sha256,
            "frozen_model_artifact_sha256": model_sha256,
            "profile_sha256": profile_sha256,
            "initial_controller_state_id": model_state["rank_state_id"],
        },
    )
    return ExactModelRuntimeBindingConfig(
        candidate_contract_path=candidate_contract_path,
        expected_candidate_contract_sha256=candidate_contract_sha256,
        frozen_model_binding_path=binding_path,
        expected_frozen_model_binding_sha256=binding_sha256,
        frozen_model_artifact_path=model_path,
        expected_frozen_model_artifact_sha256=model_sha256,
        candidate_profile_path=profile_path,
        expected_candidate_profile_sha256=profile_sha256,
    )


def test_runtime_exact_booster_profile_and_controller_bytes_pass(
    tmp_path: Path,
) -> None:
    summary = verify_exact_model_runtime_binding(
        _runtime_fixture(tmp_path)
    )
    assert summary["runtime_byte_verification_passed"] is True
    assert all(summary["checks"].values())
    assert len(summary["verified_hashes"]["booster_bytes_sha256"]) == 64
    assert summary["candidate_name"] == "candidate-from-contract"


@pytest.mark.parametrize(
    ("fixture_kwargs", "failed_check"),
    [
        (
            {"booster_binding_drift": True},
            "booster_bytes_match_frozen_contract",
        ),
        ({"profile_binding_drift": True}, "profile_file_sha256"),
        (
            {"controller_binding_drift": True},
            "initial_controller_state_exact",
        ),
    ],
)
def test_runtime_byte_mismatches_fail_closed(
    tmp_path: Path,
    fixture_kwargs: dict,
    failed_check: str,
) -> None:
    config = _runtime_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(
        ExactModelRuntimeBindingError,
        match=failed_check,
    ):
        verify_exact_model_runtime_binding(config)


def test_missing_runtime_binding_file_fails_closed(tmp_path: Path) -> None:
    config = _runtime_fixture(tmp_path)
    Path(config.frozen_model_binding_path).unlink()
    with pytest.raises(ExactModelRuntimeBindingError, match="is missing"):
        verify_exact_model_runtime_binding(config)


def test_collector_rejects_runtime_drift_before_creating_service_root(
    tmp_path: Path,
) -> None:
    config = _runtime_fixture(tmp_path, booster_binding_drift=True)
    service_root = tmp_path / "must-not-exist"
    with pytest.raises(ExactModelRuntimeBindingError):
        run_service(
            service_root=service_root,
            protocol_path=tmp_path / "not-opened-protocol.json",
            protocol_sha256="a" * 64,
            batch_round_count=12,
            max_batches=1,
            max_consecutive_failures=1,
            failure_backoff_seconds=0.0,
            challenge_candidate_contract_path=(
                config.candidate_contract_path
            ),
            challenge_candidate_contract_sha256=(
                config.expected_candidate_contract_sha256
            ),
            challenge_frozen_model_binding_path=(
                config.frozen_model_binding_path
            ),
            challenge_frozen_model_binding_sha256=(
                config.expected_frozen_model_binding_sha256
            ),
            challenge_frozen_model_artifact_path=(
                config.frozen_model_artifact_path
            ),
            challenge_frozen_model_artifact_sha256=(
                config.expected_frozen_model_artifact_sha256
            ),
            challenge_candidate_profile_path=config.candidate_profile_path,
            challenge_candidate_profile_sha256=(
                config.expected_candidate_profile_sha256
            ),
        )
    assert service_root.exists() is False
