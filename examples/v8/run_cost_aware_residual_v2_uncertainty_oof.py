"""Run or verify the sole BTC 15m residual v2 uncertainty challenger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.cost_aware_residual_v2_uncertainty import (  # noqa: E402
    DEFAULT_CHALLENGER_OUTPUT_DIR,
    DEFAULT_CHALLENGER_PROTOCOL,
    run_uncertainty_challenger_oof,
    verify_frozen_uncertainty_challenger_oof,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "verify"), required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_CHALLENGER_PROTOCOL))
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--output-dir", default=str(DEFAULT_CHALLENGER_OUTPUT_DIR))
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.mode == "run":
        if not args.protocol_sha256 or not args.source_commit:
            parser.error("--protocol-sha256 and --source-commit are required for run")
        result = run_uncertainty_challenger_oof(
            protocol_path=args.protocol,
            expected_protocol_sha256=args.protocol_sha256,
            output_dir=args.output_dir,
            source_commit=args.source_commit,
            repository_root=ROOT,
        )
    else:
        result = verify_frozen_uncertainty_challenger_oof(
            protocol_path=args.protocol,
            output_dir=args.output_dir,
            repository_root=ROOT,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
