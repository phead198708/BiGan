"""Generate or verify the BTC 15m residual two-slot terminal review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_terminal_review import (  # noqa: E402
    DEFAULT_REVIEW_PATH,
    generate_residual_terminal_review,
    verify_residual_terminal_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "verify"), required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--review", default=str(DEFAULT_REVIEW_PATH))
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.created_at:
            parser.error("--created-at is required for generate")
        result = generate_residual_terminal_review(
            created_at=args.created_at,
            repository_root=ROOT,
            output_path=args.review,
        )
    else:
        result = verify_residual_terminal_review(
            repository_root=ROOT,
            review_path=args.review,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
