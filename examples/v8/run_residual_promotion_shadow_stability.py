"""Freeze outcome-blind shadow-stability evidence after exact population freeze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_release_evidence import (  # noqa: E402
    build_outcome_blind_shadow_stability_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--population-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    report = build_outcome_blind_shadow_stability_report(
        repository_root=ROOT,
        service_root=args.service_root,
        freeze_dir=args.freeze_dir,
        expected_population_manifest_sha256=args.population_manifest_sha256,
        output_path=args.output,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
