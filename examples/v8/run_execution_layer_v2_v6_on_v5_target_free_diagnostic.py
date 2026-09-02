"""Run the issue #209 v6-on-v5 target-free viability diagnostic."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_v6_on_v5_target_free_diagnostic import (
    V6OnV5TargetFreeDiagnosticConfig,
    run_v6_on_v5_target_free_diagnostic,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/v8/polymarket_runs"))
    parser.add_argument(
        "--diagnostic-profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_v6_on_v5_target_free_diagnostic_v1.json"
        ),
    )
    parser.add_argument("--expected-diagnostic-profile-sha256", required=True)
    parser.add_argument("--v5-freeze-manifest", type=Path, required=True)
    parser.add_argument("--v6-profile", type=Path, required=True)
    parser.add_argument("--v6-preregistration-manifest", type=Path, required=True)
    parser.add_argument("--collector-index", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--freeze-created-ts", type=int)
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = _parse_args()
    result = run_v6_on_v5_target_free_diagnostic(
        V6OnV5TargetFreeDiagnosticConfig(
            run_id=args.run_id,
            output_dir=args.output_dir,
            diagnostic_profile_path=args.diagnostic_profile,
            expected_diagnostic_profile_sha256=args.expected_diagnostic_profile_sha256,
            v5_freeze_manifest_path=args.v5_freeze_manifest,
            v6_profile_path=args.v6_profile,
            v6_preregistration_manifest_path=args.v6_preregistration_manifest,
            collector_index_path=args.collector_index,
            feature_contract_path=args.feature_contract,
            implementation_commit=args.implementation_commit or _head(),
            freeze_created_ts=args.freeze_created_ts or int(time.time() * 1000),
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(
        json.dumps(
            {
                "run_dir": str(result["run_dir"]),
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "diagnostic_viability_passed": result["report"]["diagnostic_viability_passed"],
                "blocking_reason_codes": result["report"][
                    "diagnostic_viability_blocking_reason_codes"
                ],
                "full_guard_accepted_bet_count": result["report"]["full_guard_accepted_bet_count"],
                "full_guard_accepted_side_distribution": result["report"][
                    "full_guard_accepted_side_distribution"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
