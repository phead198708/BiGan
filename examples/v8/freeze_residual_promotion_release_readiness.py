"""Freeze non-authorizing residual promotion release-readiness artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_release_readiness import (  # noqa: E402
    freeze_release_readiness_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    result = freeze_release_readiness_contract(
        repository_root=ROOT,
        created_at=args.created_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
