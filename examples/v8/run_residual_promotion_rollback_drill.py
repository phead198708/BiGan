"""Run the frozen residual promotion zero-capital rollback drill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_rollback import (  # noqa: E402
    run_zero_capital_rollback_drill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "examples/v8/polymarket_configs/"
            "BTC-15M-cost-aware-market-residual-promotion-v1/"
            "zero_capital_rollback_drill_report.json"
        ),
    )
    args = parser.parse_args()
    report = run_zero_capital_rollback_drill(
        repository_root=ROOT,
        output_path=args.output,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
