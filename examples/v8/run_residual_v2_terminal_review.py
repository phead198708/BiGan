"""Generate or verify the BTC 15m residual v2 terminal review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_v2_terminal_review import (  # noqa: E402
    generate_residual_v2_terminal_review,
    verify_residual_v2_terminal_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "verify"), required=True)
    args = parser.parse_args()
    result = (
        generate_residual_v2_terminal_review(repository_root=ROOT)
        if args.mode == "generate"
        else verify_residual_v2_terminal_review(repository_root=ROOT)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
