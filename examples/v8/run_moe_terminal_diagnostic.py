"""Generate or verify the deterministic BTC 15m MoE terminal diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.moe_terminal_diagnostic import (  # noqa: E402
    DEFAULT_CONTRACT,
    DEFAULT_OUTPUT_DIR,
    generate_terminal_diagnostic,
    verify_frozen_terminal_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--mode",
        choices=("generate", "verify", "verify-from-raw"),
        default="verify",
    )
    args = parser.parse_args()
    if args.mode == "generate":
        result = generate_terminal_diagnostic(
            contract_path=args.contract,
            output_dir=args.output_dir,
            repository_root=ROOT,
        )
    else:
        result = verify_frozen_terminal_diagnostic(
            contract_path=args.contract,
            output_dir=args.output_dir,
            repository_root=ROOT,
            recompute_scored_rows_from_raw=args.mode == "verify-from-raw",
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
