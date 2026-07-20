#!/usr/bin/env python3
"""Run the frozen #227 outcome-free p_up-semantic compatibility canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    PUpSemanticCompatibilityV67Config,
    run_p_up_semantic_compatibility_v6_7,
)

DEFAULT_ROOT = Path(
    "examples/v8/polymarket_runs/"
    "policy-selected-runtime-pnl-v6-6-fresh-calibration-freeze-20260720T232530Z"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/v8/polymarket_runs")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(
            "examples/v8/polymarket_configs/"
            "execution_layer_v2_p_up_semantic_compatibility_v6_7_profile.json"
        ),
    )
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument(
        "--source-freeze-manifest",
        type=Path,
        default=DEFAULT_ROOT / "v6_6_fresh_calibration_prediction_freeze_manifest.json",
    )
    parser.add_argument(
        "--expected-source-freeze-manifest-sha256",
        default="c9634613d51199cb1ebe9cf8205230a31643dfaa01573040a527063c1f248f91",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_ROOT / "v6_2_target_free_predictions.jsonl",
    )
    parser.add_argument(
        "--expected-predictions-sha256",
        default="ae2d4763bc5f18a057cb1b6165062bc29398c292e5f251a4fa1e61f566e29e90",
    )
    parser.add_argument(
        "--five-action-rows",
        type=Path,
        default=DEFAULT_ROOT / "v6_6_target_free_five_action_rows.jsonl",
    )
    parser.add_argument(
        "--expected-five-action-rows-sha256",
        default="718151b1fa7e7f4ef59ab09c6e65b84966d5ca14450843f261a8310124edd03a",
    )
    parser.add_argument(
        "--legacy-guard-replay",
        type=Path,
        default=DEFAULT_ROOT / "v6_2_outcome_blind_guard_replay.jsonl",
    )
    parser.add_argument(
        "--expected-legacy-guard-replay-sha256",
        default="3c3bca259aef6430d53ec544a2478bd3debdc41ea00230ca49fc8fda015f7a1a",
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()
    result = run_p_up_semantic_compatibility_v6_7(
        PUpSemanticCompatibilityV67Config(
            run_id=args.run_id,
            output_dir=args.output_dir,
            profile_path=args.profile,
            expected_profile_sha256=args.expected_profile_sha256,
            source_freeze_manifest_path=args.source_freeze_manifest,
            expected_source_freeze_manifest_sha256=(
                args.expected_source_freeze_manifest_sha256
            ),
            predictions_path=args.predictions,
            expected_predictions_sha256=args.expected_predictions_sha256,
            five_action_rows_path=args.five_action_rows,
            expected_five_action_rows_sha256=args.expected_five_action_rows_sha256,
            legacy_guard_replay_path=args.legacy_guard_replay,
            expected_legacy_guard_replay_sha256=(
                args.expected_legacy_guard_replay_sha256
            ),
            implementation_commit=args.implementation_commit,
            overwrite_existing=args.overwrite_existing,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
