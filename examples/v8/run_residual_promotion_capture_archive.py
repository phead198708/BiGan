"""Mirror and verify completed promotion captures without touching outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_capture_archive import (  # noqa: E402
    mirror_completed_capture_snapshot,
    verify_capture_archive_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    mirror = commands.add_parser("mirror", help="copy and verify ledger-closed attempts")
    mirror.add_argument("--service-root", type=Path, required=True)
    mirror.add_argument("--archive-root", type=Path, required=True)
    mirror.add_argument("--created-at", default=datetime.now(UTC).isoformat())
    verify = commands.add_parser("verify", help="verify an existing frozen snapshot")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-source-root", type=Path)
    args = parser.parse_args()
    if args.command == "mirror":
        report = mirror_completed_capture_snapshot(
            service_root=args.service_root,
            archive_root=args.archive_root,
            created_at=args.created_at,
        )
    else:
        report = verify_capture_archive_snapshot(
            manifest_path=args.manifest,
            expected_source_root=args.expected_source_root,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
