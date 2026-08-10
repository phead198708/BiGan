"""Run the separately authorized residual-promotion settlement ingestion once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_settlement import (  # noqa: E402
    ingest_authorized_official_settlements,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID  # noqa: E402

CONFIG = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
EXECUTION_CONTRACT = CONFIG / "promotion_evaluation_execution_contract_v4.json"
SETTLEMENT_CONTRACT = CONFIG / "promotion_settlement_ingestion_contract.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--population-manifest-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = ingest_authorized_official_settlements(
        repository_root=ROOT,
        service_root=args.service_root,
        freeze_dir=args.freeze_dir,
        expected_population_manifest_sha256=args.population_manifest_sha256,
        execution_contract_path=EXECUTION_CONTRACT,
        expected_execution_contract_sha256=_sidecar(EXECUTION_CONTRACT),
        settlement_contract_path=SETTLEMENT_CONTRACT,
        expected_settlement_contract_sha256=_sidecar(SETTLEMENT_CONTRACT),
        authorization_path=args.authorization,
        expected_authorization_sha256=args.authorization_sha256,
        output_dir=args.output_dir,
        created_at=args.created_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _sidecar(path: Path) -> str:
    return path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    raise SystemExit(main())
