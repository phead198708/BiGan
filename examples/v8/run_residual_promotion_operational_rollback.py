"""Run a zero-capital operational rollback latency drill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_release_evidence import (  # noqa: E402
    run_outcome_blind_operational_rollback_drill,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    report = run_outcome_blind_operational_rollback_drill(
        repository_root=ROOT,
        output_path=args.output,
        created_at=args.created_at,
        iterations=args.iterations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
