"""Freeze the train-only #226 policy-selected runtime-PnL point model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_runtime_pnl_v6_6 import (
    PolicySelectedRuntimePNLV66Config,
    run_policy_selected_runtime_pnl_v6_6,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_policy_selected_runtime_pnl_v6_6_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--v6-5-train-manifest", type=Path, required=True)
    parser.add_argument("--expected-v6-5-train-manifest-sha256", required=True)
    parser.add_argument("--v6-2-target-free-guard-replay", type=Path, required=True)
    parser.add_argument(
        "--expected-v6-2-target-free-guard-replay-sha256", required=True
    )
    parser.add_argument("--external-selected-train-corpus-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verified_implementation_commit(value: str | None) -> str:
    head = _head()
    if value is not None and value != head:
        raise ValueError(
            f"implementation commit does not match current checkout HEAD: {value} != {head}"
        )
    return head


def main() -> None:
    args = _parse_args()
    result = run_policy_selected_runtime_pnl_v6_6(
        PolicySelectedRuntimePNLV66Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            v6_5_train_manifest_path=args.v6_5_train_manifest,
            expected_v6_5_train_manifest_sha256=(
                args.expected_v6_5_train_manifest_sha256
            ),
            v6_2_target_free_guard_replay_path=args.v6_2_target_free_guard_replay,
            expected_v6_2_target_free_guard_replay_sha256=(
                args.expected_v6_2_target_free_guard_replay_sha256
            ),
            external_selected_train_corpus_dir=(
                args.external_selected_train_corpus_dir
            ),
            implementation_commit=_verified_implementation_commit(
                args.implementation_commit
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "external_selected_train_corpus_dir": str(
                    result["external_selected_train_corpus_dir"]
                ),
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "model_sha256": report["model_sha256"],
                "policy_dataset_hash": report["policy_dataset_hash"],
                "split_hash": report["split_hash"],
                "population_support": report[
                    "target_free_population_support_audit"
                ],
                "cross_fit_gate_passed": report["cross_fit"][
                    "cross_fit_gate_passed"
                ],
                "point_model_freeze_gate_passed": report[
                    "point_model_freeze_gate_passed"
                ],
                "point_model_freeze_blocking_reason_codes": report[
                    "point_model_freeze_blocking_reason_codes"
                ],
                "fresh_calibration_collection_allowed": result["manifest"][
                    "fresh_calibration_collection_allowed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
