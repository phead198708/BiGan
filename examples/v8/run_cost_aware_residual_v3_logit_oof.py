"""Run the second and final SHA-frozen BTC 15m residual v3 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigan.v8.polymarket.cost_aware_residual_v3_logit import (
    DEFAULT_CHALLENGER_OUTPUT_DIR,
    DEFAULT_CHALLENGER_PROTOCOL,
    run_logit_challenger_oof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_CHALLENGER_PROTOCOL)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHALLENGER_OUTPUT_DIR)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        result = run_logit_challenger_oof(
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
