"""Offline operator CHILD fixture, never imported/instantiated by the supervisor.

    Real operator + session + ledger; deterministic public-feed-shaped samples.
    The initial window is already in progress, ending after twice tail cutoff
    (at least one second); subsequent windows retain their configured duration.
    No simulated clock, strategy overrides, forced orders or network requests.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
from contextlib import suppress
from typing import Any

from bigan.paper_trading.operator.config import OperatorConfig, load_operator_config
from bigan.paper_trading.operator.discovery import DiscoveredMarket, DiscoverySelection
from bigan.paper_trading.operator.pricing_inputs import ReferencePriceSample
from bigan.paper_trading.operator.resolution import FinalResolution
from bigan.paper_trading.operator.runtime import PaperTradingOperator

from .report import now_ms


class MockMarkets:
    def __init__(self, config: OperatorConfig) -> None:
        self.config = config
        self.first_end = now_ms() + max(1000, 2 * config.pricing_tail_cutoff_ms)

    def market(self, index: int) -> DiscoveredMarket:
        c = self.config
        end = self.first_end + index * c.window_duration_ms
        start = end - c.window_duration_ms
        return DiscoveredMarket(
            market_id=f"mock-{start}", condition_id=f"mock-condition-{start}",
            slug=f"{c.underlying.lower()}-updown-{c.window_duration_ms // 60000}m-{start}",
            title=f"{c.underlying} offline paper stack fixture", underlying=c.underlying,
            market_type=c.market_type, window_duration_ms=c.window_duration_ms,
            start_ts_ms=start, end_ts_ms=end, yes_token_id=f"mock-yes-{start}", no_token_id=f"mock-no-{start}",
            active=True, closed=False, accepting_orders=True, source_endpoint="mock://local-fixture",
            discovered_at_ms=now_ms(), resolution_source="mock-final-only",
            resolution_identity="mock-stack-v1", reference_price_at_start=100000.0, raw_payload_sha256="0" * 64,
        )

    async def discover(self, **_kwargs: Any) -> DiscoverySelection:
        index = max(0, (now_ms() - self.first_end) // self.config.window_duration_ms + 1)
        return DiscoverySelection(current=self.market(index), next=self.market(index + 1), eligible_count=2)

    async def resolve(self, market: DiscoveredMarket, **_kwargs: Any) -> FinalResolution | None:
        now = now_ms()
        if now < market.end_ts_ms:
            return None
        return FinalResolution(
            market_id=market.market_id, condition_id=market.condition_id, window_id=market.window_id,
            yes_payout=1.0, settlement_ts_ms=market.end_ts_ms, source="mock-final-only",
            source_ts_ms=market.end_ts_ms, received_ts_ms=now,
            source_reference=f"mock-resolution:{market.market_id}", resolution_identity=market.resolution_identity,
        )


class MockFeeds:
    def __init__(self, operator: PaperTradingOperator, *, hold_quotes: bool = False) -> None:
        self.operator = operator
        self.config = operator.config
        self.generation = -1
        self.sequence = 0
        self.update_id = 1
        self.bid, self.ask = 99999, 100001
        self.hold_quotes = hold_quotes

    async def tick(self) -> None:
        o, c = self.operator, self.config
        market = o.active_market
        if market is None:
            return
        now = now_ms()
        if o.generation != self.generation:
            self.generation, self.sequence, self.update_id = o.generation, 0, 1
            self.bid, self.ask = 99999, 100001
            samples = max(c.volatility_min_samples + 1, c.ofi_min_samples + 1)
            base = now - samples * c.volatility_return_interval_ms
            await o.ingest_binance_snapshot({"lastUpdateId": 1, "bids": [[str(self.bid), "2"]],
                                            "asks": [[str(self.ask), "2"]]},
                                           generation=self.generation, received_at_ms=base)
            for index in range(samples):
                await self._delta(base + (index + 1) * c.volatility_return_interval_ms)
            await o.ingest_oracle(self._oracle(max(market.start_ts_ms, now - c.twap_window_ms)),
                                  generation=self.generation)
        else:
            await self._delta(now)
        await o.ingest_oracle(self._oracle(now), generation=self.generation)
        self.sequence += 1
        for token in (market.yes_token_id, market.no_token_id):
            await o.ingest_market_message({
                "event_type": "book", "sequence": self.sequence, "timestamp": now,
                "asset_id": token, "bids": [{"price": "0.99" if self.hold_quotes else "0.49", "size": "100"}],
                "asks": [{"price": "1.0" if self.hold_quotes else "0.51", "size": "100"}],
            }, generation=self.generation, received_at_ms=now)

    async def _delta(self, timestamp: int) -> None:
        self.update_id += 1
        bid = 100010 + self.update_id % 17
        ask = bid + 2
        await self.operator.ingest_binance_delta({
            "s": self.config.binance_symbol, "E": timestamp, "U": self.update_id, "u": self.update_id,
            "b": [[str(self.bid), "0"], [str(bid), "2"]],
            "a": [[str(self.ask), "0"], [str(ask), "2"]],
        }, generation=self.generation, received_at_ms=timestamp)
        self.bid, self.ask = bid, ask

    def _oracle(self, timestamp: int) -> ReferencePriceSample:
        return ReferencePriceSample(timestamp_ms=timestamp, received_at_ms=now_ms(), price=100000.0,
                                    source=f"polymarket_rtds_chainlink:{self.config.chainlink_symbol.lower()}")


async def run(config: OperatorConfig, *, hold_quotes: bool = False) -> None:
    markets = MockMarkets(config)
    operator = PaperTradingOperator(config=config, discovery=markets, resolution=markets, clock_ms=now_ms)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    feeds = MockFeeds(operator, hold_quotes=hold_quotes)
    try:
        await operator.start()
        while not stop.is_set():
            if operator.state.value in {"FAILED", "EXHAUSTED"}:
                raise RuntimeError("mock operator cannot continue")
            await feeds.tick()
            await operator.poll()
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=0.25)
    finally:
        await operator.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline paper operator child fixture")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--hold-quotes", action="store_true", help="Offline expensive-quote fixture, not a strategy override")
    args = parser.parse_args(argv)
    try:
        config = load_operator_config(args.config)
        if not config.mock or config.config_check_only or config.config_sha256 != args.expected_config_sha256:
            raise ValueError("invalid mock configuration")
        asyncio.run(run(config, hold_quotes=args.hold_quotes))
    except Exception:
        print("mock operator failed (details suppressed)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
