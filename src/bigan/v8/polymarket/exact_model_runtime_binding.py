"""Fail-closed runtime verification for the exact v8.5 challenger bytes."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256

RUNTIME_BINDING_SCHEMA_VERSION = (
    "bigan-v8-exact-model-runtime-byte-verification-v1"
)


class ExactModelRuntimeBindingError(ValueError):
    """Raised before collection or scoring when frozen runtime bytes drift."""


@dataclass(frozen=True, slots=True)
class ExactModelRuntimeBindingConfig:
    """Paths and raw-file pins needed at decision-runtime startup."""

    candidate_contract_path: Path | str
    expected_candidate_contract_sha256: str
    frozen_model_binding_path: Path | str
    expected_frozen_model_binding_sha256: str
    frozen_model_artifact_path: Path | str
    expected_frozen_model_artifact_sha256: str
    candidate_profile_path: Path | str
    expected_candidate_profile_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "candidate_contract_path",
            "frozen_model_binding_path",
            "frozen_model_artifact_path",
            "candidate_profile_path",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)))
        for field in (
            "expected_candidate_contract_sha256",
            "expected_frozen_model_binding_sha256",
            "expected_frozen_model_artifact_sha256",
            "expected_candidate_profile_sha256",
        ):
            value = str(getattr(self, field)).lower()
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field} must be a SHA-256 hex digest")
            object.__setattr__(self, field, value)


def verify_exact_model_runtime_binding(
    config: ExactModelRuntimeBindingConfig,
) -> dict[str, Any]:
    """Load and verify the exact booster, profile, and controller state."""

    candidate_contract_bytes = _required_bytes(
        config.candidate_contract_path,
        "candidate contract",
    )
    binding_bytes = _required_bytes(
        config.frozen_model_binding_path,
        "frozen model binding",
    )
    model_bytes = _required_bytes(
        config.frozen_model_artifact_path,
        "frozen model artifact",
    )
    profile_bytes = _required_bytes(
        config.candidate_profile_path,
        "candidate profile",
    )
    candidate_contract = _json_bytes(
        candidate_contract_bytes,
        "candidate contract",
    )
    binding = _json_bytes(binding_bytes, "frozen model binding")
    model = _json_bytes(model_bytes, "frozen model artifact")
    profile = _json_bytes(profile_bytes, "candidate profile")
    weighted_model = dict(model.get("final_weighted_model") or {})
    loaded_state = dict(model.get("final_rank_state") or {})
    expected_state = dict(binding.get("initial_controller_state") or {})
    loaded_bound_state = _runtime_controller_binding_state(loaded_state)
    candidate_name = candidate_contract.get("primary_policy")

    booster_bytes: bytes | None = None
    booster_load_succeeded = False
    encoded_booster = weighted_model.get("booster_json_base64")
    if isinstance(encoded_booster, str) and encoded_booster:
        try:
            booster_bytes = base64.b64decode(
                encoded_booster,
                validate=True,
            )
            booster = xgb.Booster()
            booster.load_model(bytearray(booster_bytes))
            booster_load_succeeded = True
        except (binascii.Error, ValueError, xgb.core.XGBoostError):
            booster_bytes = None

    booster_contract_sha256 = (
        canonical_json_sha256(
            base64.b64encode(booster_bytes).decode("ascii")
        )
        if booster_bytes is not None
        else ""
    )
    booster_bytes_sha256 = (
        hashlib.sha256(booster_bytes).hexdigest()
        if booster_bytes is not None
        else ""
    )
    actual = {
        "candidate_contract_file_sha256": _sha256(candidate_contract_bytes),
        "binding_file_sha256": _sha256(binding_bytes),
        "model_artifact_file_sha256": _sha256(model_bytes),
        "profile_file_sha256": _sha256(profile_bytes),
        "booster_contract_sha256": booster_contract_sha256,
        "booster_bytes_sha256": booster_bytes_sha256,
        "initial_controller_state_payload_sha256": (
            canonical_json_sha256(loaded_bound_state)
            if loaded_bound_state
            else ""
        ),
        "rank_state_id": str(loaded_state.get("rank_state_id") or ""),
    }
    checks = {
        "candidate_contract_file_sha256": (
            actual["candidate_contract_file_sha256"]
            == config.expected_candidate_contract_sha256
        ),
        "binding_file_sha256": (
            actual["binding_file_sha256"]
            == config.expected_frozen_model_binding_sha256
            and candidate_contract.get("frozen_model_binding_sha256")
            == actual["binding_file_sha256"]
        ),
        "model_artifact_file_sha256": (
            actual["model_artifact_file_sha256"]
            == config.expected_frozen_model_artifact_sha256
            and candidate_contract.get("frozen_model_artifact_sha256")
            == actual["model_artifact_file_sha256"]
            and binding.get("frozen_model_artifact_sha256")
            == actual["model_artifact_file_sha256"]
        ),
        "profile_file_sha256": (
            actual["profile_file_sha256"]
            == config.expected_candidate_profile_sha256
            and candidate_contract.get("profile_sha256")
            == actual["profile_file_sha256"]
            and binding.get("frozen_profile_sha256")
            == actual["profile_file_sha256"]
        ),
        "candidate_identity_from_contract": (
            isinstance(candidate_name, str)
            and bool(candidate_name)
            and binding.get("candidate_name") == candidate_name
            and model.get("candidate_name") == candidate_name
            and profile.get("candidate_name") == candidate_name
        ),
        "booster_bytes_loaded": (
            booster_load_succeeded
            and booster_bytes is not None
            and bool(booster_bytes)
        ),
        "booster_bytes_match_frozen_contract": (
            booster_contract_sha256
            == binding.get("frozen_booster_sha256")
            == weighted_model.get("booster_sha256")
        ),
        "initial_controller_state_exact": (
            bool(loaded_bound_state)
            and loaded_bound_state == expected_state
        ),
        "initial_controller_rank_state_id": (
            _valid_sha256(actual["rank_state_id"])
            and actual["rank_state_id"]
            == expected_state.get("rank_state_id")
            == candidate_contract.get("initial_controller_state_id")
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": RUNTIME_BINDING_SCHEMA_VERSION,
        "candidate_name": candidate_name,
        "checks": checks,
        "verified_hashes": actual,
        "booster_hash_semantics": (
            "canonical_sha256_of_base64_encoded_exact_loaded_booster_bytes"
        ),
        "profile_hash_semantics": "sha256_of_exact_loaded_profile_file_bytes",
        "controller_state_semantics": (
            "exact_payload_equality_plus_frozen_rank_state_id"
        ),
        "runtime_byte_verification_passed": passed,
        "paper_candidate_allowed": False,
        "live_trading_enabled": False,
        "capital_at_risk": False,
        "polymarket_write_enabled": False,
        "wallet_signing_enabled": False,
        "v8_execution_handoff_allowed": False,
        "source_model_candidate_eligible": False,
        "freeze_ready": False,
        "promotion_evidence_eligible": False,
        "#134_resume_allowed": False,
        "#146_start_allowed": False,
    }
    summary["summary_id"] = canonical_json_sha256(summary)
    blockers = [name for name, value in checks.items() if not value]
    if blockers:
        raise ExactModelRuntimeBindingError(
            "exact model runtime byte verification failed: "
            + ",".join(blockers)
        )
    return summary


def validate_runtime_binding_summary(summary: dict[str, Any]) -> None:
    """Reject caller-invented or mutated runtime verification summaries."""

    if not isinstance(summary, dict):
        raise ExactModelRuntimeBindingError(
            "runtime byte verification summary must be an object"
        )
    checks = summary.get("checks")
    verified_hashes = summary.get("verified_hashes")
    without_id = {
        key: value for key, value in summary.items() if key != "summary_id"
    }
    safety_false = (
        "paper_candidate_allowed",
        "live_trading_enabled",
        "capital_at_risk",
        "polymarket_write_enabled",
        "wallet_signing_enabled",
        "v8_execution_handoff_allowed",
        "source_model_candidate_eligible",
        "freeze_ready",
        "promotion_evidence_eligible",
        "#134_resume_allowed",
        "#146_start_allowed",
    )
    valid = (
        summary.get("schema_version") == RUNTIME_BINDING_SCHEMA_VERSION
        and isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values())
        and isinstance(verified_hashes, dict)
        and bool(verified_hashes)
        and all(_valid_sha256(value) for value in verified_hashes.values())
        and summary.get("runtime_byte_verification_passed") is True
        and all(summary.get(field) is False for field in safety_false)
        and summary.get("summary_id") == canonical_json_sha256(without_id)
    )
    if not valid:
        raise ExactModelRuntimeBindingError(
            "runtime byte verification summary is invalid"
        )


def _required_bytes(path_value: Path | str, label: str) -> bytes:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ExactModelRuntimeBindingError(f"{label} file is missing: {path}")
    return path.read_bytes()


def _runtime_controller_binding_state(
    loaded_state: dict[str, Any],
) -> dict[str, Any]:
    controller = dict(loaded_state.get("next_controller_decision") or {})
    return {
        "rank_state_id": loaded_state.get("rank_state_id"),
        "rank_lineage_hash": loaded_state.get("rank_lineage_hash"),
        "eligible_prediction_scores_hash": loaded_state.get(
            "eligible_prediction_scores_hash"
        ),
        "controller_guard_acceptance_history_hash": loaded_state.get(
            "controller_guard_acceptance_history_hash"
        ),
        "controller_state_uses_target_outcome_label_or_pnl": (
            not (
                loaded_state.get(
                    "controller_target_outcome_label_or_pnl_free"
                )
                is True
                and loaded_state.get(
                    "rank_state_uses_target_outcome_or_pnl"
                )
                is False
            )
        ),
        "future_controller_updates_use_strictly_prior_guard_results_only": (
            controller.get("controller_source")
            == "strictly_prior_full_guard_acceptance_only"
            and controller.get("current_market_guard_result_used") is False
            and controller.get("target_outcome_label_or_pnl_used") is False
        ),
    }


def _json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactModelRuntimeBindingError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ExactModelRuntimeBindingError(
            f"{label} must be a JSON object"
        )
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()) is not None


__all__ = [
    "RUNTIME_BINDING_SCHEMA_VERSION",
    "ExactModelRuntimeBindingConfig",
    "ExactModelRuntimeBindingError",
    "validate_runtime_binding_summary",
    "verify_exact_model_runtime_binding",
]
