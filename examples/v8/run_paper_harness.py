"""Run the deterministic v8 paper trading harness.

This example is paper-only: it writes synthetic paper orders, fills, ledger
entries, paper observations, Phase 5 safety evidence, and Phase 6 paper-mode
CI/CD evidence. It never places real orders or touches live capital.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bigan.v8.paper import (  # noqa: E402
    PaperDegradationConfig,
    PaperHarnessConfig,
    run_paper_trading_harness,
    synthetic_phase4_decisions,
)

DEFAULT_RUN_ID = "paper_harness_synthetic_v1"
FIXED_CREATED_AT = "2026-06-22T01:00:00Z"


def run_paper_harness(
    output_dir: Path | str,
    *,
    run_id: str = DEFAULT_RUN_ID,
    inject_degradation: bool = False,
) -> dict[str, object]:
    """Run a deterministic paper-only harness and return a concise summary."""

    config = PaperHarnessConfig(
        run_id=run_id,
        candidate_run_id="paper-candidate-001",
        model_sha256=_sha256_text("paper-model"),
        policy_dataset_hash=_sha256_text("paper-policy-dataset"),
        split_hash=_sha256_text("paper-split"),
        upstream_training_report_sha256=_sha256_text("paper-training-report"),
        upstream_validation_report_sha256=_sha256_text("paper-validation-report"),
        output_dir=Path(output_dir) / run_id,
        created_at=FIXED_CREATED_AT,
        degradation=(
            PaperDegradationConfig(
                start_index=4,
                net_return_shift=0.035,
                cost_multiplier=5.0,
                live_regime="high_volatility",
            )
            if inject_degradation
            else None
        ),
    )
    result = run_paper_trading_harness(
        decisions=synthetic_phase4_decisions(),
        config=config,
    )
    return {
        "paper_bundle_manifest": str(result.artifact_paths["paper_bundle_manifest"]),
        "paper_only": result.bundle_manifest["paper_only"],
        "capital_at_risk": result.bundle_manifest["capital_at_risk"],
        "phase5_passed": result.phase5_result.passed,
        "phase5_kill_switch_triggered": (
            result.phase5_result.report.safety_action["kill_switch_triggered"]
        ),
        "phase6_deployment_status": result.phase6_result.report.deployment_status,
        "paper_order_stream_sha256": result.paper_report.paper_order_stream_sha256,
        "paper_fill_stream_sha256": result.paper_report.paper_fill_stream_sha256,
        "paper_ledger_sha256": result.paper_report.paper_ledger_sha256,
        "paper_positions_sha256": result.paper_report.paper_positions_sha256,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "v8" / "artifacts",
        help="Directory that will contain the run-scoped paper artifact bundle.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--inject-degradation",
        action="store_true",
        help="Inject deterministic paper degradation to exercise the Phase 5 kill-switch.",
    )
    args = parser.parse_args(argv)
    summary = run_paper_harness(
        args.output_dir,
        run_id=args.run_id,
        inject_degradation=args.inject_degradation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
