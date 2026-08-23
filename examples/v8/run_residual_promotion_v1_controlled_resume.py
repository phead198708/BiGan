"""Resume the frozen promotion collector from one exact authorized boundary."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket.residual_promotion_controlled_resume import (  # noqa: E402
    validate_controlled_resume_authorization,
)
from bigan.v8.polymarket.residual_promotion_migrated_restart import (  # noqa: E402
    verify_existing_coverage_resume_record,
)

CONFIG = (
    ROOT
    / "examples/v8/polymarket_configs"
    / "BTC-15M-cost-aware-market-residual-promotion-v1"
)
ORIGINAL_COLLECTOR = ROOT / "examples/v8/run_residual_promotion_v1_collector.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument(
        "--resume-authorization",
        type=Path,
        default=CONFIG / "controlled_resume_authorization_20260817.json",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    service_root = args.service_root.expanduser().resolve()
    controlled = validate_controlled_resume_authorization(
        authorization_path=args.resume_authorization,
        repository_root=ROOT,
        service_root=service_root,
    )
    original = _load_original_collector()
    original_args = original._parser().parse_args(
        ["--service-root", str(service_root)]
    )
    validation = original.validate_collection_authorization(
        authorization_path=original_args.authorization,
        collector_protocol_path=original_args.collector_protocol,
        repository_root=ROOT,
    )
    authorization = controlled["authorization"]
    if not (
        validation["authorization_sha256"]
        == authorization["frozen_collection_authorization_sha256"]
        and validation["collector_protocol_sha256"]
        == authorization["frozen_collector_protocol_sha256"]
        and validation["bundle"]["sha256"]
        == authorization["frozen_candidate_bundle_sha256"]
    ):
        raise ValueError("controlled resume frozen lineage binding drift")

    def _verified_start_record(
        root: Path,
        *,
        validation: dict,
        resumed_attempt_count: int,
        rest_fallback_collection_seconds: float,
    ) -> None:
        if resumed_attempt_count != controlled["attempts_consumed"]:
            raise ValueError("controlled resume attempt boundary changed before start")
        verify_existing_coverage_resume_record(
            root,
            original_validation=validation,
            resumed_attempt_count=resumed_attempt_count,
            rest_fallback_collection_seconds=rest_fallback_collection_seconds,
        )

    original._write_start_record = _verified_start_record
    runtime = validation["runtime"]
    baseline = original.load_matched_baseline(repository_root=ROOT)
    chainlink = original.PolymarketChainlinkRTDSCollector()
    chainlink.start()
    try:
        return original._collect(
            args=original_args,
            root=service_root,
            validation=validation,
            runtime=runtime,
            baseline=baseline,
            chainlink=chainlink,
        )
    finally:
        chainlink.stop()


def _load_original_collector() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "frozen_residual_promotion_v1_collector", ORIGINAL_COLLECTOR
    )
    if spec is None or spec.loader is None:
        raise ValueError("original frozen collector cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
