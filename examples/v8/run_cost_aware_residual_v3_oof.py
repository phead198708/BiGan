"""Run one SHA-frozen BTC 15m causal time-adaptive residual v3 OOF slot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.cost_aware_residual_v3 import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROTOCOL,
    run_residual_v3_rolling_origin_oof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        result = run_residual_v3_rolling_origin_oof(
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
