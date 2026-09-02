"""Migrate and verify a stopped promotion-v1 collection service root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_service_root_migration import (  # noqa: E402
    migrate_service_root,
    verify_service_root_migration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser("migrate")
    migrate.add_argument("--source-root", type=Path, required=True)
    migrate.add_argument("--destination-root", type=Path, required=True)
    migrate.add_argument("--report", type=Path, required=True)
    migrate.add_argument("--created-at", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--skip-source", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "migrate":
        report = migrate_service_root(
            source_root=args.source_root,
            destination_root=args.destination_root,
            report_path=args.report,
            created_at=args.created_at,
        )
    else:
        report = verify_service_root_migration(
            report_path=args.report,
            require_source_match=not args.skip_source,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
