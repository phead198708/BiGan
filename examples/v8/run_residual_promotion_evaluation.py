"""Dry-run or execute the frozen residual-promotion evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.challenge_development_lane import sha256_file  # noqa: E402
from bigan.v8.polymarket.residual_promotion_evaluation import (  # noqa: E402
    dry_run_evaluation_pipeline,
    run_authorized_promotion_evaluation,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID  # noqa: E402

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dry-run")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--service-root", type=Path, required=True)
    evaluate.add_argument("--freeze-dir", type=Path, required=True)
    evaluate.add_argument("--population-manifest-sha256", required=True)
    evaluate.add_argument("--settlements", type=Path, required=True)
    evaluate.add_argument("--settlements-sha256", required=True)
    evaluate.add_argument("--authorization", type=Path, required=True)
    evaluate.add_argument("--authorization-sha256", required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--created-at")
    return parser


def main() -> int:
    args = _parser().parse_args()
    contract = CONFIG / "promotion_evaluation_execution_contract.json"
    if args.command == "dry-run":
        protocol = json.loads(
            (CONFIG / "prospective_statistical_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        result = dry_run_evaluation_pipeline(protocol=protocol)
    else:
        result = run_authorized_promotion_evaluation(
            repository_root=ROOT,
            service_root=args.service_root,
            freeze_dir=args.freeze_dir,
            expected_population_manifest_sha256=(
                args.population_manifest_sha256
            ),
            settlements_path=args.settlements,
            expected_settlements_sha256=args.settlements_sha256,
            execution_contract_path=contract,
            expected_execution_contract_sha256=sha256_file(contract),
            authorization_path=args.authorization,
            expected_authorization_sha256=args.authorization_sha256,
            output_dir=args.output_dir,
            created_at=args.created_at,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
