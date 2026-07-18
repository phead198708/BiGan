"""Run the frozen #191 prospective accepted-bet power analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_accepted_bet_power import (  # noqa: E402
    run_pairwise_accepted_bet_power_analysis,
)

DEFAULT_DESIGN = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_accepted_bet_power_v1.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="examples/v8/polymarket_runs")
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--design-sha256", required=True)
    args = parser.parse_args(argv)
    result = run_pairwise_accepted_bet_power_analysis(
        run_id=args.run_id,
        output_dir=args.output_dir,
        design_path=args.design,
        expected_design_sha256=args.design_sha256,
    )
    report = result["report"]
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "power_analysis_ready": report["power_analysis_ready"],
                "recommended_required_accepted_unique_market_count": report[
                    "recommended_required_accepted_unique_market_count"
                ],
                "recommended_quality_valid_market_count": report[
                    "recommended_quality_valid_market_count"
                ],
                "recommended_maximum_capture_attempt_count": report[
                    "recommended_maximum_capture_attempt_count"
                ],
                "uses_current_oof_validation_or_confirmatory_pnl": False,
                "report_path": str(result["report_path"]),
                "report_sha256": result["report_sha256"],
                "manifest_path": str(result["manifest_path"]),
                "manifest_sha256": result["manifest_sha256"],
                "paper_only": True,
                "capital_at_risk": False,
                "v8_execution_handoff_allowed": False,
                "#134_resume_allowed": False,
                "#146_start_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
