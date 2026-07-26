"""Write a launchd descriptor for the persistent outcome-blind collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples/v8/run_execution_layer_v2_persistent_outcome_blind_collector.py"
DEFAULT_PROTOCOL = ROOT / (
    "examples/v8/polymarket_configs/execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
DEFAULT_BATCH_CANARY_FEATURE_CONTRACT = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256 = (
    "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
)


def write_launchd_plist(
    *,
    output_path: Path | str,
    label: str,
    service_root: Path | str,
    protocol_path: Path | str,
    protocol_sha256: str,
    batch_round_count: int,
    python_executable: Path | str,
    max_batches: int = 0,
    batch_canary_feature_contract_path: Path | str = DEFAULT_BATCH_CANARY_FEATURE_CONTRACT,
    batch_canary_feature_contract_sha256: str = (DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256),
    v6_2_candidate_manifest_path: Path | str | None = None,
    v6_2_candidate_manifest_sha256: str | None = None,
    v6_6_point_freeze_manifest_path: Path | str | None = None,
    v6_6_point_freeze_manifest_sha256: str | None = None,
    v6_9_candidate_manifest_path: Path | str | None = None,
    v6_9_candidate_manifest_sha256: str | None = None,
    v6_9_collection_plan_path: Path | str | None = None,
    v6_9_collection_plan_sha256: str | None = None,
    require_challenge_runtime_binding: bool = False,
    challenge_candidate_contract_path: Path | str | None = None,
    challenge_candidate_contract_sha256: str | None = None,
    challenge_frozen_model_binding_path: Path | str | None = None,
    challenge_frozen_model_binding_sha256: str | None = None,
    challenge_frozen_model_artifact_path: Path | str | None = None,
    challenge_frozen_model_artifact_sha256: str | None = None,
    challenge_candidate_profile_path: Path | str | None = None,
    challenge_candidate_profile_sha256: str | None = None,
) -> dict:
    """Write a KeepAlive service descriptor without starting the service."""

    if not label.strip():
        raise ValueError("launchd label is required")
    if batch_round_count <= 0:
        raise ValueError("batch_round_count must be positive")
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    service_root = Path(service_root).expanduser().resolve()
    direct_training_root = Path("/Volumes/PHILIPS/v8")
    if service_root == direct_training_root or direct_training_root in service_root.parents:
        raise ValueError("raw collector service_root cannot use direct training corpus root")
    protocol_path = Path(protocol_path).expanduser().resolve()
    if _sha256(protocol_path) != protocol_sha256.lower():
        raise ValueError("persistent collector protocol SHA-256 mismatch")
    batch_canary_feature_contract_path = (
        Path(batch_canary_feature_contract_path).expanduser().resolve()
    )
    if _sha256(batch_canary_feature_contract_path) != batch_canary_feature_contract_sha256.lower():
        raise ValueError("batch canary feature contract SHA-256 mismatch")
    if (v6_2_candidate_manifest_path is None) != (v6_2_candidate_manifest_sha256 is None):
        raise ValueError("v6.2 candidate manifest path and SHA-256 must be provided together")
    if v6_2_candidate_manifest_path is not None:
        v6_2_candidate_manifest_path = Path(v6_2_candidate_manifest_path).expanduser().resolve()
        if _sha256(v6_2_candidate_manifest_path) != str(v6_2_candidate_manifest_sha256).lower():
            raise ValueError("v6.2 candidate manifest SHA-256 mismatch")
    if (v6_6_point_freeze_manifest_path is None) != (v6_6_point_freeze_manifest_sha256 is None):
        raise ValueError("v6.6 point freeze manifest path and SHA-256 must be provided together")
    if v6_6_point_freeze_manifest_path is not None:
        v6_6_point_freeze_manifest_path = (
            Path(v6_6_point_freeze_manifest_path).expanduser().resolve()
        )
        if (
            _sha256(v6_6_point_freeze_manifest_path)
            != str(v6_6_point_freeze_manifest_sha256).lower()
        ):
            raise ValueError("v6.6 point freeze manifest SHA-256 mismatch")
    v6_9_values = (
        v6_9_candidate_manifest_path,
        v6_9_candidate_manifest_sha256,
        v6_9_collection_plan_path,
        v6_9_collection_plan_sha256,
    )
    if any(value is not None for value in v6_9_values) and any(
        value is None for value in v6_9_values
    ):
        raise ValueError("v6.9 candidate and collection-plan paths/hashes are required together")
    if v6_9_candidate_manifest_path is not None:
        if v6_2_candidate_manifest_path is None:
            raise ValueError("v6.9 launchd mode requires the frozen v6.2 candidate")
        v6_9_candidate_manifest_path = Path(v6_9_candidate_manifest_path).expanduser().resolve()
        v6_9_collection_plan_path = Path(v6_9_collection_plan_path).expanduser().resolve()
        if _sha256(v6_9_candidate_manifest_path) != str(v6_9_candidate_manifest_sha256).lower():
            raise ValueError("v6.9 candidate manifest SHA-256 mismatch")
        if _sha256(v6_9_collection_plan_path) != str(v6_9_collection_plan_sha256).lower():
            raise ValueError("v6.9 collection plan SHA-256 mismatch")
    challenge_values = (
        challenge_candidate_contract_path,
        challenge_candidate_contract_sha256,
        challenge_frozen_model_binding_path,
        challenge_frozen_model_binding_sha256,
        challenge_frozen_model_artifact_path,
        challenge_frozen_model_artifact_sha256,
        challenge_candidate_profile_path,
        challenge_candidate_profile_sha256,
    )
    if any(value is not None for value in challenge_values) and any(
        value is None for value in challenge_values
    ):
        raise ValueError(
            "challenge contract, binding, model, and profile paths/hashes "
            "are required together"
        )
    if require_challenge_runtime_binding and not all(
        value is not None for value in challenge_values
    ):
        raise ValueError("challenge runtime byte binding is required")
    if challenge_candidate_contract_path is not None:
        challenge_pairs = (
            (
                "candidate contract",
                challenge_candidate_contract_path,
                challenge_candidate_contract_sha256,
            ),
            (
                "frozen model binding",
                challenge_frozen_model_binding_path,
                challenge_frozen_model_binding_sha256,
            ),
            (
                "frozen model artifact",
                challenge_frozen_model_artifact_path,
                challenge_frozen_model_artifact_sha256,
            ),
            (
                "candidate profile",
                challenge_candidate_profile_path,
                challenge_candidate_profile_sha256,
            ),
        )
        normalized_challenge_paths: dict[str, Path] = {}
        for label_name, path_value, expected_hash in challenge_pairs:
            resolved = Path(path_value).expanduser().resolve()
            if _sha256(resolved) != str(expected_hash).lower():
                raise ValueError(
                    f"challenge {label_name} SHA-256 mismatch"
                )
            normalized_challenge_paths[label_name] = resolved
        challenge_candidate_contract_path = normalized_challenge_paths[
            "candidate contract"
        ]
        challenge_frozen_model_binding_path = normalized_challenge_paths[
            "frozen model binding"
        ]
        challenge_frozen_model_artifact_path = normalized_challenge_paths[
            "frozen model artifact"
        ]
        challenge_candidate_profile_path = normalized_challenge_paths[
            "candidate profile"
        ]
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    service_root.mkdir(parents=True, exist_ok=True)
    program_arguments = [
        str(Path(python_executable).expanduser().resolve()),
        str(RUNNER),
        "--service-root",
        str(service_root),
        "--protocol",
        str(protocol_path),
        "--protocol-sha256",
        protocol_sha256.lower(),
        "--batch-round-count",
        str(batch_round_count),
        "--batch-canary-feature-contract",
        str(batch_canary_feature_contract_path),
        "--batch-canary-feature-contract-sha256",
        batch_canary_feature_contract_sha256.lower(),
        "--max-batches",
        str(max_batches),
    ]
    if v6_2_candidate_manifest_path is not None:
        program_arguments.extend(
            [
                "--v6-2-candidate-manifest",
                str(v6_2_candidate_manifest_path),
                "--v6-2-candidate-manifest-sha256",
                str(v6_2_candidate_manifest_sha256).lower(),
            ]
        )
    if v6_6_point_freeze_manifest_path is not None:
        program_arguments.extend(
            [
                "--v6-6-point-freeze-manifest",
                str(v6_6_point_freeze_manifest_path),
                "--v6-6-point-freeze-manifest-sha256",
                str(v6_6_point_freeze_manifest_sha256).lower(),
            ]
        )
    if v6_9_candidate_manifest_path is not None:
        program_arguments.extend(
            [
                "--v6-9-candidate-manifest",
                str(v6_9_candidate_manifest_path),
                "--v6-9-candidate-manifest-sha256",
                str(v6_9_candidate_manifest_sha256).lower(),
                "--v6-9-collection-plan",
                str(v6_9_collection_plan_path),
                "--v6-9-collection-plan-sha256",
                str(v6_9_collection_plan_sha256).lower(),
            ]
        )
    if challenge_candidate_contract_path is not None:
        program_arguments.extend(
            [
                "--challenge-candidate-contract",
                str(challenge_candidate_contract_path),
                "--challenge-candidate-contract-sha256",
                str(challenge_candidate_contract_sha256).lower(),
                "--challenge-frozen-model-binding",
                str(challenge_frozen_model_binding_path),
                "--challenge-frozen-model-binding-sha256",
                str(challenge_frozen_model_binding_sha256).lower(),
                "--challenge-frozen-model-artifact",
                str(challenge_frozen_model_artifact_path),
                "--challenge-frozen-model-artifact-sha256",
                str(challenge_frozen_model_artifact_sha256).lower(),
                "--challenge-candidate-profile",
                str(challenge_candidate_profile_path),
                "--challenge-candidate-profile-sha256",
                str(challenge_candidate_profile_sha256).lower(),
            ]
        )
    payload = {
        "Label": label,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": {
            "PYTHONPATH": "src:.",
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(service_root / "persistent_collector_stdout.log"),
        "StandardErrorPath": str(service_root / "persistent_collector_stderr.log"),
    }
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
        handle.flush()
    temporary.replace(output_path)
    return {
        "launchd_plist_path": str(output_path),
        "launchd_plist_sha256": _sha256(output_path),
        "launchd_label": label,
        "service_root": str(service_root),
        "continuous_collection": max_batches == 0,
        "maximum_batch_count": max_batches,
        "restart_supervision_enabled": True,
        "automatic_outcome_blind_batch_canary_enabled": True,
        "automatic_v6_2_frozen_batch_canary_enabled": (v6_2_candidate_manifest_path is not None),
        "v6_2_candidate_manifest_path": (
            str(v6_2_candidate_manifest_path) if v6_2_candidate_manifest_path is not None else None
        ),
        "v6_2_candidate_manifest_sha256": (
            str(v6_2_candidate_manifest_sha256).lower()
            if v6_2_candidate_manifest_sha256 is not None
            else None
        ),
        "v6_6_fresh_calibration_collection_mode": (v6_6_point_freeze_manifest_path is not None),
        "v6_6_point_freeze_manifest_path": (
            str(v6_6_point_freeze_manifest_path)
            if v6_6_point_freeze_manifest_path is not None
            else None
        ),
        "v6_6_point_freeze_manifest_sha256": (
            str(v6_6_point_freeze_manifest_sha256).lower()
            if v6_6_point_freeze_manifest_sha256 is not None
            else None
        ),
        "automatic_v6_9_frozen_batch_canary_enabled": (v6_9_candidate_manifest_path is not None),
        "v6_9_candidate_manifest_path": (
            str(v6_9_candidate_manifest_path) if v6_9_candidate_manifest_path is not None else None
        ),
        "v6_9_candidate_manifest_sha256": (
            str(v6_9_candidate_manifest_sha256).lower()
            if v6_9_candidate_manifest_sha256 is not None
            else None
        ),
        "v6_9_collection_plan_path": (
            str(v6_9_collection_plan_path) if v6_9_collection_plan_path is not None else None
        ),
        "v6_9_collection_plan_sha256": (
            str(v6_9_collection_plan_sha256).lower()
            if v6_9_collection_plan_sha256 is not None
            else None
        ),
        "challenge_runtime_byte_binding_required": (
            require_challenge_runtime_binding
        ),
        "challenge_runtime_byte_binding_configured": (
            challenge_candidate_contract_path is not None
        ),
        "challenge_candidate_contract_path": (
            str(challenge_candidate_contract_path)
            if challenge_candidate_contract_path is not None
            else None
        ),
        "challenge_candidate_contract_sha256": (
            str(challenge_candidate_contract_sha256).lower()
            if challenge_candidate_contract_sha256 is not None
            else None
        ),
        "challenge_frozen_model_binding_path": (
            str(challenge_frozen_model_binding_path)
            if challenge_frozen_model_binding_path is not None
            else None
        ),
        "challenge_frozen_model_binding_sha256": (
            str(challenge_frozen_model_binding_sha256).lower()
            if challenge_frozen_model_binding_sha256 is not None
            else None
        ),
        "challenge_frozen_model_artifact_path": (
            str(challenge_frozen_model_artifact_path)
            if challenge_frozen_model_artifact_path is not None
            else None
        ),
        "challenge_frozen_model_artifact_sha256": (
            str(challenge_frozen_model_artifact_sha256).lower()
            if challenge_frozen_model_artifact_sha256 is not None
            else None
        ),
        "challenge_candidate_profile_path": (
            str(challenge_candidate_profile_path)
            if challenge_candidate_profile_path is not None
            else None
        ),
        "challenge_candidate_profile_sha256": (
            str(challenge_candidate_profile_sha256).lower()
            if challenge_candidate_profile_sha256 is not None
            else None
        ),
        "batch_canary_feature_contract_path": str(batch_canary_feature_contract_path),
        "batch_canary_feature_contract_sha256": batch_canary_feature_contract_sha256.lower(),
        "outcome_blind_collection_only": True,
        "settlement_finalizer_started": False,
        "resolution_provider_called": False,
        "training_corpus_export_attempted": False,
        "labels_outcomes_or_pnl_opened": False,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--label",
        default="com.bigan.v8.persistent-outcome-blind-collector",
    )
    parser.add_argument("--service-root", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--batch-round-count", type=int, default=12)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop successfully after this many batches; zero runs continuously.",
    )
    parser.add_argument(
        "--batch-canary-feature-contract",
        default=str(DEFAULT_BATCH_CANARY_FEATURE_CONTRACT),
    )
    parser.add_argument(
        "--batch-canary-feature-contract-sha256",
        default=DEFAULT_BATCH_CANARY_FEATURE_CONTRACT_SHA256,
    )
    parser.add_argument("--v6-2-candidate-manifest")
    parser.add_argument("--v6-2-candidate-manifest-sha256")
    parser.add_argument("--v6-6-point-freeze-manifest")
    parser.add_argument("--v6-6-point-freeze-manifest-sha256")
    parser.add_argument("--v6-9-candidate-manifest")
    parser.add_argument("--v6-9-candidate-manifest-sha256")
    parser.add_argument("--v6-9-collection-plan")
    parser.add_argument("--v6-9-collection-plan-sha256")
    parser.add_argument(
        "--require-challenge-runtime-binding",
        action="store_true",
    )
    parser.add_argument("--challenge-candidate-contract")
    parser.add_argument("--challenge-candidate-contract-sha256")
    parser.add_argument("--challenge-frozen-model-binding")
    parser.add_argument("--challenge-frozen-model-binding-sha256")
    parser.add_argument("--challenge-frozen-model-artifact")
    parser.add_argument("--challenge-frozen-model-artifact-sha256")
    parser.add_argument("--challenge-candidate-profile")
    parser.add_argument("--challenge-candidate-profile-sha256")
    parser.add_argument("--python-executable", default=sys.executable)
    args = parser.parse_args(argv)
    report = write_launchd_plist(
        output_path=args.output_path,
        label=args.label,
        service_root=args.service_root,
        protocol_path=args.protocol,
        protocol_sha256=args.protocol_sha256,
        batch_round_count=args.batch_round_count,
        python_executable=args.python_executable,
        max_batches=args.max_batches,
        batch_canary_feature_contract_path=args.batch_canary_feature_contract,
        batch_canary_feature_contract_sha256=args.batch_canary_feature_contract_sha256,
        v6_2_candidate_manifest_path=args.v6_2_candidate_manifest,
        v6_2_candidate_manifest_sha256=args.v6_2_candidate_manifest_sha256,
        v6_6_point_freeze_manifest_path=args.v6_6_point_freeze_manifest,
        v6_6_point_freeze_manifest_sha256=args.v6_6_point_freeze_manifest_sha256,
        v6_9_candidate_manifest_path=args.v6_9_candidate_manifest,
        v6_9_candidate_manifest_sha256=args.v6_9_candidate_manifest_sha256,
        v6_9_collection_plan_path=args.v6_9_collection_plan,
        v6_9_collection_plan_sha256=args.v6_9_collection_plan_sha256,
        require_challenge_runtime_binding=(
            args.require_challenge_runtime_binding
        ),
        challenge_candidate_contract_path=(
            args.challenge_candidate_contract
        ),
        challenge_candidate_contract_sha256=(
            args.challenge_candidate_contract_sha256
        ),
        challenge_frozen_model_binding_path=(
            args.challenge_frozen_model_binding
        ),
        challenge_frozen_model_binding_sha256=(
            args.challenge_frozen_model_binding_sha256
        ),
        challenge_frozen_model_artifact_path=(
            args.challenge_frozen_model_artifact
        ),
        challenge_frozen_model_artifact_sha256=(
            args.challenge_frozen_model_artifact_sha256
        ),
        challenge_candidate_profile_path=args.challenge_candidate_profile,
        challenge_candidate_profile_sha256=(
            args.challenge_candidate_profile_sha256
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
