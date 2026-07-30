"""Generate outcome-blind BTC 15m MoE collection monitoring artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.moe_collection_observability import (  # noqa: E402
    build_collection_observability,
    build_development_distribution_reference,
    build_development_distribution_shift_reference,
    build_evaluation_dry_run_report,
    build_finalization_checklist,
)

DEFAULT_SERVICE_ROOT = ROOT / (
    "examples/v8/polymarket_runs/"
    "BTC-15M-MoE-confirmatory-v2-outcome-blind-collection-001"
)
DEFAULT_CONFIG_ROOT = ROOT / (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", default=str(DEFAULT_SERVICE_ROOT))
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--build-development-reference", action="store_true")
    parser.add_argument(
        "--build-development-distribution-shift-reference",
        action="store_true",
    )
    parser.add_argument("--skip-live-monitor", action="store_true")
    args = parser.parse_args()
    config_root = Path(args.config_root).resolve()
    output: dict[str, object] = {}
    if args.build_development_reference:
        output["development_distribution_reference"] = (
            build_development_distribution_reference(
                output_path=(
                    config_root / "moe_development_distribution_reference.json"
                ),
                repository_root=ROOT,
            )
        )
    if args.build_development_distribution_shift_reference:
        output["development_distribution_shift_reference"] = (
            build_development_distribution_shift_reference(
                output_path=(
                    config_root
                    / "moe_development_distribution_shift_reference.json"
                ),
                repository_root=ROOT,
            )
        )
    if not args.skip_live_monitor:
        output["live_monitor"] = build_collection_observability(
            service_root=args.service_root,
            repository_root=ROOT,
        )
    output["dry_run"] = build_evaluation_dry_run_report(
        output_path=config_root / "moe_confirmatory_evaluation_dry_run_report.json",
        repository_root=ROOT,
    )
    output["finalization_checklist"] = build_finalization_checklist(
        output_path=config_root / "moe_confirmatory_finalization_checklist.json",
    )
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
