"""CLI for the paper-only operator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from contextlib import suppress
from pathlib import Path

from bigan.build_provenance import BuildProvenanceError, require_source_commit

from .config import OperatorConfig, load_operator_config
from .discovery import DiscoveredMarket, DiscoverySelection
from .live import LiveFeedSupervisor
from .pricing_inputs import ReferencePriceSample
from .resolution import FinalResolution, GammaResolutionClient
from .runtime import PaperTradingOperator
from .transports import AiohttpPublicJSONClient, GammaDiscoveryClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BiGan paper-only trading operator")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="validate and print safe identity")
    parser.add_argument("--mock-demo", action="store_true", help="run one local no-network demo")
    parser.add_argument("--expected-config-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-commit", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_operator_config(args.config)
    if args.expected_config_sha256 and config.config_sha256 != args.expected_config_sha256:
        parser.error("Operator configuration identity changed before startup")
    check_only = args.check or config.config_check_only or (config.dry_run and not args.mock_demo)
    run_live = not check_only and not (args.mock_demo or config.mock)
    if run_live or args.expected_source_commit:
        try:
            if args.expected_source_commit and config.source_commit != args.expected_source_commit:
                raise BuildProvenanceError("source identity changed")
            # Standalone and supervised live execution share the same gate.
            require_source_commit(config.source_commit)
        except BuildProvenanceError:
            parser.error("Operator build provenance does not match expected source")
    logging.basicConfig(
        level=getattr(logging, config.logging_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if check_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "paper_only": True,
                    "config_sha256": config.config_sha256,
                    "config": config.config_identity(),
                },
                default=str,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if args.mock_demo or config.mock:
        asyncio.run(_run_mock_demo(config))
    else:
        asyncio.run(_run_live(config))
    return 0


async def _run_live(config: OperatorConfig) -> None:
    http = AiohttpPublicJSONClient()
    operator = PaperTradingOperator(
        config=config,
        discovery=GammaDiscoveryClient(endpoint=config.gamma_markets_endpoint, http=http),
        resolution=GammaResolutionClient(endpoint=config.resolution_endpoint, http=http),
        clock_ms=_now_ms,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)
    await LiveFeedSupervisor(operator=operator, http=http).run(stop_event)


async def _run_mock_demo(config: OperatorConfig) -> None:
    now_ms = _now_ms()
    start = now_ms - (now_ms % config.window_duration_ms)
    market = DiscoveredMarket(
        market_id=f"mock-{start}",
        condition_id=f"mock-condition-{start}",
        slug=f"{config.underlying.lower()}-updown-{config.window_duration_ms // 60_000}m-{start}",
        title=f"{config.underlying} deterministic local paper demo",
        underlying=config.underlying,
        market_type=config.market_type,
        window_duration_ms=config.window_duration_ms,
        start_ts_ms=start,
        end_ts_ms=start + config.window_duration_ms,
        yes_token_id="mock-yes-token",
        no_token_id="mock-no-token",
        active=True,
        closed=False,
        accepting_orders=True,
        source_endpoint="mock://local-fixture",
        discovered_at_ms=now_ms,
        resolution_source="mock-final-only",
        resolution_identity="mock-resolution-v1",
        reference_price_at_start=100_000.0,
        raw_payload_sha256="0" * 64,
    )
    operator = PaperTradingOperator(
        config=config,
        discovery=_MockDiscovery(market),
        resolution=_MockResolution(),
        clock_ms=_now_ms,
    )
    await operator.start()
    generation = operator.generation
    received = _now_ms()
    warmup_samples = max(config.volatility_min_samples + 1, config.ofi_min_samples)
    sample_base = received - warmup_samples * config.volatility_return_interval_ms
    await operator.ingest_binance_snapshot(
        {
            "lastUpdateId": 1,
            "bids": [["99999", "2"]],
            "asks": [["100001", "2"]],
        },
        generation=generation,
        received_at_ms=sample_base,
    )
    previous_bid, previous_ask = 99_999, 100_001
    for index in range(warmup_samples):
        timestamp = sample_base + (index + 1) * config.volatility_return_interval_ms
        next_bid, next_ask = 100_000 + index, 100_002 + index
        await operator.ingest_binance_delta(
            {
                "s": config.binance_symbol,
                "E": timestamp,
                "U": index + 2,
                "u": index + 2,
                "b": [[str(previous_bid), "0"], [str(next_bid), "2"]],
                "a": [[str(previous_ask), "0"], [str(next_ask), "2"]],
            },
            generation=generation,
            received_at_ms=timestamp,
        )
        previous_bid, previous_ask = next_bid, next_ask
    await operator.ingest_oracle(
        _mock_oracle(config, received),
        generation=generation,
    )
    for token, bid, ask in (
        (market.yes_token_id, "0.49", "0.51"),
        (market.no_token_id, "0.49", "0.51"),
    ):
        await operator.ingest_market_message(
            {
                "event_type": "book",
                "sequence": 1,
                "timestamp": received,
                "asset_id": token,
                "bids": [{"price": bid, "size": "100"}],
                "asks": [{"price": ask, "size": "100"}],
            },
            generation=generation,
            received_at_ms=received,
        )
    await operator.shutdown()
    print(
        json.dumps(
            operator.status().to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


class _MockDiscovery:
    def __init__(self, market: DiscoveredMarket) -> None:
        self.market = market

    async def discover(self, **_kwargs: object) -> DiscoverySelection:
        return DiscoverySelection(current=self.market, next=None, eligible_count=1)


class _MockResolution:
    async def resolve(self, *_args: object, **_kwargs: object) -> FinalResolution | None:
        return None


def _mock_oracle(config: OperatorConfig, timestamp_ms: int) -> ReferencePriceSample:
    return ReferencePriceSample(
        timestamp_ms=timestamp_ms,
        received_at_ms=timestamp_ms,
        price=100_000.0,
        source=f"polymarket_rtds_chainlink:{config.chainlink_symbol.lower()}",
    )


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


if __name__ == "__main__":
    raise SystemExit(main())
