#!/usr/bin/env python3
"""Monitor and freeze the exact-model challenge future window."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.challenge_future_freeze import (
    ChallengeFutureFreezeConfig,
    challenge_collection_status,
    resolve_challenge_collection_service_root,
    run_challenge_future_target_free_freeze,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _load_json,
    _sha256_file,
)

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent
CONFIG_DIR = EXAMPLES_DIR / "polymarket_configs"
DEFAULT_PLAN = CONFIG_DIR / "parallel_future_collection_plan.json"
DEFAULT_PROTOCOL = CONFIG_DIR / "parallel_candidate_protocol.json"
DEFAULT_V8_1_CONTRACT = (
    CONFIG_DIR
    / "parallel_candidate_v8_1_primary_no_fallback_contract.json"
)
DEFAULT_V8_3_CONTRACT = (
    CONFIG_DIR
    / "parallel_candidate_v8_3_primary_with_fallback_contract.json"
)
DEFAULT_V6_7_CONTRACT = (
    CONFIG_DIR
    / "parallel_candidate_matched_frozen_v6_7_contract.json"
)
DEFAULT_BINDING = CONFIG_DIR / "parallel_frozen_v8_1_model_binding.json"
DEFAULT_HISTORICAL_GATE = (
    CONFIG_DIR / "historical_replay_superiority_contract.json"
)
DEFAULT_HISTORICAL_REPORT = (
    CONFIG_DIR / "historical_replay_superiority_report.json"
)
DEFAULT_PREFREEZE_CHECKLIST = (
    CONFIG_DIR / "challenge_prefreeze_checklist.json"
)
DEFAULT_EXCLUDED_CAPTURE_LEDGER = (
    CONFIG_DIR / "challenge_prefreeze_excluded_capture_ledger.json"
)
DEFAULT_SUPERSESSION_GOVERNANCE = (
    CONFIG_DIR / "challenge_supersession_governance.json"
)
DEFAULT_COLLECTOR_PROTOCOL = (
    CONFIG_DIR
    / "execution_layer_v2_persistent_outcome_blind_collector_v1.json"
)
DEFAULT_FEATURE_CONTRACT = (
    CONFIG_DIR
    / "execution_layer_v2_pairwise_action_advantage_lcb_feature_contract_v1.json"
)
DEFAULT_V8_3_PROFILE = (
    CONFIG_DIR
    / "execution_layer_v2_non_risk_abstention_fallback_v8_3_profile.json"
)
DEFAULT_OUTPUT_DIR = EXAMPLES_DIR / "polymarket_runs"


def _git_head_and_clean() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(
            "challenge freeze requires a clean committed implementation"
        )
    return head


def _expected_artifact_sha256(
    path: Path,
    *,
    artifact_name: str,
) -> str:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise ValueError(f"{artifact_name} sidecar missing: {sidecar}")
    value = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if _sha256_file(path) != value.lower():
        raise ValueError(f"{artifact_name} SHA-256 sidecar mismatch")
    return value.lower()


def _expected_plan_sha256(path: Path) -> str:
    return _expected_artifact_sha256(
        path,
        artifact_name="collection plan",
    )


def _status(args: argparse.Namespace) -> dict:
    plan_path = args.plan.resolve()
    _expected_plan_sha256(plan_path)
    plan = _load_json(plan_path)
    service_root = resolve_challenge_collection_service_root(
        collection_plan=plan,
        collection_plan_path=plan_path,
        requested_service_root=args.service_root,
    )
    return challenge_collection_status(
        service_root=service_root,
        collection_plan=plan,
    )


def _freeze(args: argparse.Namespace) -> dict:
    implementation_commit = _git_head_and_clean()
    plan_path = args.plan.resolve()
    plan = _load_json(plan_path)
    service_root = resolve_challenge_collection_service_root(
        collection_plan=plan,
        collection_plan_path=plan_path,
        requested_service_root=args.service_root,
    )
    binding = _load_json(args.binding.resolve())
    candidate_contracts = {
        "v8_1_primary_no_fallback": _load_json(
            args.v8_1_contract.resolve()
        ),
        "v8_3_primary_with_fallback": _load_json(
            args.v8_3_contract.resolve()
        ),
        "matched_frozen_v6_7": _load_json(
            args.v6_7_contract.resolve()
        ),
    }
    config = ChallengeFutureFreezeConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        service_root=service_root,
        collection_plan_path=plan_path,
        expected_collection_plan_sha256=_expected_plan_sha256(plan_path),
        parallel_protocol_path=args.protocol,
        expected_parallel_protocol_sha256=str(
            plan["lineage"]["parallel_candidate_protocol_sha256"]
        ),
        v8_1_contract_path=args.v8_1_contract,
        expected_v8_1_contract_sha256=str(
            plan["lineage"]["candidate_contract_sha256s"][
                "v8_1_primary_no_fallback"
            ]
        ),
        v8_3_contract_path=args.v8_3_contract,
        expected_v8_3_contract_sha256=str(
            plan["lineage"]["candidate_contract_sha256s"][
                "v8_3_primary_with_fallback"
            ]
        ),
        v6_7_contract_path=args.v6_7_contract,
        expected_v6_7_contract_sha256=str(
            plan["lineage"]["candidate_contract_sha256s"][
                "matched_frozen_v6_7"
            ]
        ),
        frozen_model_binding_path=args.binding,
        expected_frozen_model_binding_sha256=str(
            plan["lineage"]["frozen_model_binding_sha256"]
        ),
        historical_gate_contract_path=args.historical_gate_contract,
        expected_historical_gate_contract_sha256=str(
            plan["historical_replay_prerequisite"][
                "gate_contract_sha256"
            ]
        ),
        historical_replay_report_path=args.historical_replay_report,
        expected_historical_replay_report_sha256=str(
            plan["historical_replay_prerequisite"]["report_sha256"]
        ),
        prefreeze_checklist_path=args.prefreeze_checklist,
        expected_prefreeze_checklist_sha256=str(
            plan["lineage"]["prefreeze_checklist_sha256"]
        ),
        excluded_capture_ledger_path=args.excluded_capture_ledger,
        expected_excluded_capture_ledger_sha256=str(
            plan["lineage"]["excluded_capture_ledger_sha256"]
        ),
        supersession_governance_path=args.supersession_governance,
        expected_supersession_governance_sha256=(
            _expected_artifact_sha256(
                args.supersession_governance.resolve(),
                artifact_name="supersession governance",
            )
        ),
        historical_fit_manifest_path=args.historical_fit_manifest,
        expected_historical_fit_manifest_sha256=str(
            binding["historical_fit_manifest_sha256"]
        ),
        collector_protocol_path=args.collector_protocol,
        expected_collector_protocol_sha256=str(
            plan["lineage"]["persistent_collector_protocol_sha256"]
        ),
        feature_contract_path=args.feature_contract,
        expected_feature_contract_sha256=str(
            plan["lineage"]["feature_contract_sha256"]
        ),
        v8_3_profile_path=args.v8_3_profile,
        expected_v8_3_profile_sha256=str(
            candidate_contracts["v8_3_primary_with_fallback"][
                "profile_sha256"
            ]
        ),
        implementation_commit=implementation_commit,
        decision_freeze_created_ts=(
            args.decision_freeze_created_ts
            if args.decision_freeze_created_ts is not None
            else time.time_ns() // 1_000_000
        ),
        overwrite_existing=args.overwrite_existing,
    )
    return run_challenge_future_target_free_freeze(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor or freeze the exact-120 outcome-blind challenge window."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status",
        help="Read collector progress without writing or opening targets.",
    )
    status.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    status.add_argument(
        "--service-root",
        type=Path,
        help=(
            "Optional exact collector root override; when omitted it is "
            "derived from the hash-pinned collection plan."
        ),
    )

    freeze = subparsers.add_parser(
        "freeze",
        help="Score and hash-freeze all three candidates before settlement.",
    )
    freeze.add_argument("--run-id", required=True)
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    freeze.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    freeze.add_argument(
        "--service-root",
        type=Path,
        help=(
            "Optional exact collector root override; when omitted it is "
            "derived from the hash-pinned collection plan."
        ),
    )
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument(
        "--v8-1-contract",
        type=Path,
        default=DEFAULT_V8_1_CONTRACT,
    )
    freeze.add_argument(
        "--v8-3-contract",
        type=Path,
        default=DEFAULT_V8_3_CONTRACT,
    )
    freeze.add_argument(
        "--v6-7-contract",
        type=Path,
        default=DEFAULT_V6_7_CONTRACT,
    )
    freeze.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    freeze.add_argument(
        "--historical-gate-contract",
        type=Path,
        default=DEFAULT_HISTORICAL_GATE,
    )
    freeze.add_argument(
        "--historical-replay-report",
        type=Path,
        default=DEFAULT_HISTORICAL_REPORT,
    )
    freeze.add_argument(
        "--prefreeze-checklist",
        type=Path,
        default=DEFAULT_PREFREEZE_CHECKLIST,
    )
    freeze.add_argument(
        "--excluded-capture-ledger",
        type=Path,
        default=DEFAULT_EXCLUDED_CAPTURE_LEDGER,
    )
    freeze.add_argument(
        "--supersession-governance",
        type=Path,
        default=DEFAULT_SUPERSESSION_GOVERNANCE,
    )
    freeze.add_argument(
        "--historical-fit-manifest",
        type=Path,
        required=True,
        help=(
            "Exact historical v8.1 manifest whose raw SHA-256 is pinned "
            "by the frozen model binding."
        ),
    )
    freeze.add_argument(
        "--collector-protocol",
        type=Path,
        default=DEFAULT_COLLECTOR_PROTOCOL,
    )
    freeze.add_argument(
        "--feature-contract",
        type=Path,
        default=DEFAULT_FEATURE_CONTRACT,
    )
    freeze.add_argument(
        "--v8-3-profile",
        type=Path,
        default=DEFAULT_V8_3_PROFILE,
    )
    freeze.add_argument("--decision-freeze-created-ts", type=int)
    freeze.add_argument("--overwrite-existing", action="store_true")

    args = parser.parse_args(argv)
    result = _status(args) if args.command == "status" else _freeze(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
