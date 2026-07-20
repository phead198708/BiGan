"""Fit and freeze the preregistered #224 runtime-aligned v6.4 candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_fit import (
    RuntimeAlignedSBCNetReturnV64FitConfig,
    run_runtime_aligned_sbc_net_return_v6_4_fit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--fit-profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_fit_profile.json"
        ),
    )
    parser.add_argument("--expected-fit-profile-sha256", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--expected-target-manifest-sha256", required=True)
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
    result = run_runtime_aligned_sbc_net_return_v6_4_fit(
        RuntimeAlignedSBCNetReturnV64FitConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            fit_profile_path=args.fit_profile,
            expected_fit_profile_sha256=args.expected_fit_profile_sha256,
            target_manifest_path=args.target_manifest,
            expected_target_manifest_sha256=args.expected_target_manifest_sha256,
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
                "candidate_manifest_path": str(result["candidate_manifest_path"]),
                "candidate_manifest_sha256": result["candidate_manifest_sha256"],
                "model_sha256": report["model_sha256"],
                "policy_dataset_hash": report["policy_dataset_hash"],
                "split_hash": report["split_hash"],
                "calibration_gate_passed": report["calibration_gate_passed"],
                "candidate_freeze_gate_passed": report[
                    "candidate_freeze_gate_passed"
                ],
                "candidate_freeze_blocking_reason_codes": report[
                    "candidate_freeze_blocking_reason_codes"
                ],
                "positive_lcb_unique_market_count": report[
                    "positive_lcb_unique_market_count"
                ],
                "positive_lcb_unique_market_count_by_side": report[
                    "positive_lcb_unique_market_count_by_side"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
