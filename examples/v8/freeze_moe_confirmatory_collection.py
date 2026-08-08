"""Freeze the completed BTC-15M-MoE-confirmatory-v2 collection population."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.moe_collection_finalization import (  # noqa: E402
    freeze_exact_confirmatory_collection,
)

DEFAULT_SERVICE_ROOT = ROOT / (
    "examples/v8/polymarket_runs/"
    "BTC-15M-MoE-confirmatory-v2-outcome-blind-collection-001"
)
DEFAULT_OUTPUT_DIR = ROOT / (
    "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2/"
    "confirmatory_collection_freeze_001"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", default=str(DEFAULT_SERVICE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    result = freeze_exact_confirmatory_collection(
        service_root=args.service_root,
        output_dir=args.output_dir,
        repository_root=ROOT,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
