"""Freeze the completed promotion-v1 population without opening outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_finalization import (  # noqa: E402
    freeze_exact_outcome_blind_population,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID  # noqa: E402

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--created-at")
    parser.add_argument(
        "--authorization",
        type=Path,
        default=CONFIG / "manual_collection_authorization_v2.json",
    )
    parser.add_argument(
        "--collector-protocol",
        type=Path,
        default=CONFIG / "prospective_collector_protocol_v2.json",
    )
    args = parser.parse_args()
    report = freeze_exact_outcome_blind_population(
        service_root=args.service_root,
        repository_root=ROOT,
        authorization_path=args.authorization,
        collector_protocol_path=args.collector_protocol,
        output_dir=args.output_dir,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
