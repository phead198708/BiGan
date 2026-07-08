#!/usr/bin/env python3
"""Run v8 Execution Layer v2 HTS regime risk replay diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from bigan.v8.polymarket.training.execution_layer_v2_policy_replay import (
    ExecutionLayerV2HTSRegimeRiskReplayConfig,
    run_execution_layer_v2_hts_regime_risk_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay diagnostic-only HTS regime risk policies from a settled "
            "paper-goal run directory, settlement CSV, or JSONL."
        ),
    )
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="examples/v8/polymarket_runs",
        type=Path,
    )
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    try:
        result = run_execution_layer_v2_hts_regime_risk_replay(
            ExecutionLayerV2HTSRegimeRiskReplayConfig(
                run_id=args.run_id,
                input_path=args.input_path,
                output_dir=args.output_dir,
                overwrite_existing=args.overwrite_existing,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    report = result.report
    print(f"run_id={args.run_id}")
    print(f"output_dir={result.output_dir}")
    print(f"fill_count={report['fill_count']}")
    print(f"hts_fill_count={report['hts_fill_count']}")
    print(f"up_hts_fill_count={report['up_hts_fill_count']}")
    print(
        "baseline_all_pnl="
        f"{report['policy_variants']['baseline_all']['settled_pnl']:.6f}"
    )
    print(
        "side_blind_hts_pnl="
        f"{report['policy_variants']['side_blind_hts']['settled_pnl']:.6f}"
    )
    print(
        "regime_aware_up_down_hts_pnl="
        f"{report['policy_variants']['regime_aware_up_down_hts']['settled_pnl']:.6f}"
    )
    print(
        "global_up_hts_disable_recommended="
        f"{str(report['global_up_hts_disable_recommended']).lower()}"
    )
    print(
        "report_path="
        f"{result.artifact_paths['execution_layer_v2_hts_regime_risk_replay_report']}"
    )
    print(
        "report_sha256="
        f"{result.artifact_hashes['execution_layer_v2_hts_regime_risk_replay_report']}"
    )
    print(
        "manifest_path="
        f"{result.artifact_paths['execution_layer_v2_hts_regime_risk_replay_manifest']}"
    )
    print(
        "manifest_sha256="
        f"{result.artifact_hashes['execution_layer_v2_hts_regime_risk_replay_manifest']}"
    )
    print("paper_only=true")
    print("capital_at_risk=false")
    print("v8_execution_handoff_allowed=false")


if __name__ == "__main__":
    main()
