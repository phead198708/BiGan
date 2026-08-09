"""Build, verify, or restore the BTC 15m MoE raw capture recovery bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.raw_capture_archive import (  # noqa: E402
    DEFAULT_ARCHIVE_PARENT,
    DEFAULT_INDEX,
    build_recovered_capture_archive,
    inventory_recovered_capture_archive,
    restore_recovered_capture_archive,
    verify_recovered_capture_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inventory", "build", "verify", "restore"), required=True)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--archive-parent", default=str(DEFAULT_ARCHIVE_PARENT))
    parser.add_argument("--bundle-dir")
    parser.add_argument("--shallow", action="store_true")
    args = parser.parse_args()
    if args.mode == "inventory":
        result = inventory_recovered_capture_archive(
            index_path=args.index,
            repository_root=ROOT,
        )
        result = {key: value for key, value in result.items() if key != "entries"}
    elif args.mode == "build":
        result = build_recovered_capture_archive(
            index_path=args.index,
            archive_parent=args.archive_parent,
            repository_root=ROOT,
        )
    else:
        if not args.bundle_dir:
            parser.error("--bundle-dir is required for verify or restore")
        if args.mode == "verify":
            result = verify_recovered_capture_archive(
                bundle_dir=args.bundle_dir,
                repository_root=ROOT,
                deep=not args.shallow,
            )
        else:
            result = restore_recovered_capture_archive(
                bundle_dir=args.bundle_dir,
                repository_root=ROOT,
            )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
