"""Freeze lineage or build the #224 runtime-aligned SBC target corpus."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    RuntimeAlignedSBCNetReturnV64Config,
    run_runtime_aligned_sbc_net_return_v6_4,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("freeze_lineage", "build_labels"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--issue-223-lineage-manifest", type=Path, required=True)
    parser.add_argument("--v6-2-historical-manifest", type=Path, required=True)
    parser.add_argument("--external-corpus-dir", type=Path)
    parser.add_argument("--lineage-freeze-manifest", type=Path)
    parser.add_argument("--expected-lineage-freeze-manifest-sha256")
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
    result = run_runtime_aligned_sbc_net_return_v6_4(
        RuntimeAlignedSBCNetReturnV64Config(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            issue_223_lineage_manifest_path=args.issue_223_lineage_manifest,
            v6_2_historical_manifest_path=args.v6_2_historical_manifest,
            implementation_commit=_verified_implementation_commit(
                args.implementation_commit
            ),
            external_corpus_dir=args.external_corpus_dir,
            lineage_freeze_manifest_path=args.lineage_freeze_manifest,
            expected_lineage_freeze_manifest_sha256=(
                args.expected_lineage_freeze_manifest_sha256
            ),
            overwrite_existing=args.overwrite_existing,
        )
    )
    if args.stage == "freeze_lineage":
        payload = {
            "run_dir": str(result["run_dir"]),
            "lineage_manifest_path": str(result["lineage_manifest_path"]),
            "lineage_manifest_sha256": result["lineage_manifest_sha256"],
            "market_count": result["manifest"]["market_count"],
            "lineage_freeze_passed": result["manifest"]["lineage_freeze_passed"],
        }
    else:
        payload = {
            "run_dir": str(result["run_dir"]),
            "external_corpus_dir": str(result["external_corpus_dir"]),
            "target_manifest_path": str(result["target_manifest_path"]),
            "target_manifest_sha256": result["target_manifest_sha256"],
            "market_count": result["report"]["market_count"],
            "target_row_count": result["report"]["target_row_count"],
            "position_lifecycle_class_counts": result["report"][
                "position_lifecycle_class_counts"
            ],
            "target_corpus_gate_passed": result["report"][
                "target_corpus_gate_passed"
            ],
            "target_corpus_gate_reason_codes": result["report"][
                "target_corpus_gate_reason_codes"
            ],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
