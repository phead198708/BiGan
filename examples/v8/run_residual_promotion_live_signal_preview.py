#!/usr/bin/env python3
"""Publish frozen-candidate signals during each running BTC 15m round."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bigan.v8.polymarket import (  # noqa: E402
    PolymarketChainlinkRTDSCollector,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
)
from bigan.v8.polymarket.residual_promotion_collection import (  # noqa: E402
    validate_collection_authorization,
)
from bigan.v8.polymarket.residual_promotion_live_signal_preview import (  # noqa: E402
    run_live_preview_monitor,
)
from bigan.v8.polymarket.residual_promotion_v1 import LINEAGE_ID  # noqa: E402

CONFIG_ROOT = ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=CONFIG_ROOT / "manual_collection_authorization_v3.json",
    )
    parser.add_argument(
        "--collector-protocol",
        type=Path,
        default=CONFIG_ROOT / "prospective_collector_protocol_v3.json",
    )
    parser.add_argument("--snapshot-lead-seconds", type=float, default=3.0)
    args = parser.parse_args()

    service_root = args.service_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(service_root) or service_root.is_relative_to(output):
        raise ValueError("live preview output must be disjoint from service root")
    validation = validate_collection_authorization(
        authorization_path=args.authorization,
        collector_protocol_path=args.collector_protocol,
        repository_root=ROOT,
    )
    runtime = validation["runtime"]
    bundle_sha256 = str(validation["bundle"]["sha256"])

    def provider_factory(slug: str) -> PolymarketPublicHTTPRealCorpusProvider:
        return PolymarketPublicHTTPRealCorpusProvider(
            market_slugs=(slug,),
            max_markets=1,
            timeout_seconds=20.0,
            http_timeout_seconds=20.0,
            use_rest_orderbooks=True,
        )

    def config_factory(slug: str) -> PolymarketRealCorpusRecorderConfig:
        return PolymarketRealCorpusRecorderConfig(
            run_id=f"{LINEAGE_ID}-live-preview-{slug}",
            output_dir=output.parent / "live_preview_never_collection_input",
            market_families=("btc_updown_15m",),
            mock_public_data=False,
            overwrite_existing=False,
        )

    chainlink = PolymarketChainlinkRTDSCollector()
    chainlink.start()
    try:
        run_live_preview_monitor(
            output_path=output,
            runtime=runtime,
            candidate_bundle_sha256=bundle_sha256,
            provider_factory=provider_factory,
            config_factory=config_factory,
            chainlink=chainlink,
            snapshot_lead_seconds=args.snapshot_lead_seconds,
        )
    finally:
        chainlink.stop()


if __name__ == "__main__":
    main()
