"""Dynamic spot/oracle/TWAP/volatility provider tests."""

from __future__ import annotations

import math

import pytest

from bigan.paper_trading.operator.pricing_inputs import (
    ReferencePriceSample,
    RollingPricingInputsProvider,
)


def _provider(*, max_samples: int = 10, min_returns: int = 2) -> RollingPricingInputsProvider:
    return RollingPricingInputsProvider(
        window_start_ts_ms=0,
        window_end_ts_ms=10_000,
        spot_source="binance:BTCUSDT",
        oracle_source="polymarket-rtds:btc/usd",
        max_age_ms=1_000,
        max_samples=max_samples,
        twap_window_ms=10_000,
        return_interval_ms=1_000,
        volatility_window_ms=10_000,
        volatility_min_samples=min_returns,
        volatility_max_abs_log_return=0.20,
        annualization_seconds=31_536_000,
    )


def _sample(ts_ms: int, price: float, source: str) -> ReferencePriceSample:
    return ReferencePriceSample(
        timestamp_ms=ts_ms,
        received_at_ms=ts_ms + 1,
        price=price,
        source=source,
    )


def test_provider_requires_both_sources_and_minimum_volatility_samples() -> None:
    provider = _provider()
    provider.ingest_spot(_sample(1_000, 100.0, "binance:BTCUSDT"))
    provider.ingest_oracle(_sample(1_000, 99.0, "polymarket-rtds:btc/usd"))
    assert provider(1_000) is None

    provider.ingest_spot(_sample(2_000, 101.0, "binance:BTCUSDT"))
    provider.ingest_oracle(_sample(2_000, 101.0, "polymarket-rtds:btc/usd"))
    assert provider(2_000) is None

    provider.ingest_spot(_sample(3_000, 100.5, "binance:BTCUSDT"))
    provider.ingest_oracle(_sample(3_000, 100.0, "polymarket-rtds:btc/usd"))
    inputs = provider(3_000)

    assert inputs is not None
    assert inputs.timestamp_ms == 3_000
    assert inputs.spot_price == 100.5
    assert inputs.oracle_twap_so_far == pytest.approx(100.0)
    assert inputs.twap_weight == pytest.approx(0.3)
    assert inputs.volatility_annualized > 0.0
    assert math.isfinite(inputs.volatility_annualized)


def test_future_stale_wrong_source_and_outlier_inputs_fail_closed() -> None:
    provider = _provider(min_returns=1)
    assert provider.ingest_spot(_sample(1_000, 100.0, "wrong")) is False
    assert provider.source_mismatch_count == 1
    assert provider.ingest_spot(_sample(1_000, 100.0, "binance:BTCUSDT"))
    assert provider.ingest_oracle(
        _sample(1_000, 100.0, "polymarket-rtds:btc/usd")
    )
    assert provider.ingest_spot(_sample(2_000, 200.0, "binance:BTCUSDT")) is False
    assert provider.outlier_count == 1

    assert provider.ingest_spot(_sample(2_000, 101.0, "binance:BTCUSDT"))
    assert provider.ingest_oracle(
        _sample(2_000, 101.0, "polymarket-rtds:btc/usd")
    )
    assert provider(1_999) is None
    assert provider.future_input_count == 1
    assert provider(3_001) is None
    assert provider.stale_input_count == 1


def test_buffers_are_bounded_and_reconnect_requires_warmup_again() -> None:
    provider = _provider(max_samples=3, min_returns=1)
    for index in range(1, 7):
        provider.ingest_spot(
            _sample(index * 1_000, 100.0 + index, "binance:BTCUSDT")
        )
        provider.ingest_oracle(
            _sample(index * 1_000, 99.0 + index, "polymarket-rtds:btc/usd")
        )

    assert provider.spot_sample_count == 3
    assert provider.oracle_sample_count == 3
    assert provider.return_sample_count <= 3
    assert provider(6_000) is not None

    provider.reset_for_reconnect()
    assert provider.spot_sample_count == 0
    assert provider.oracle_sample_count == 0
    assert provider.return_sample_count == 0
    assert provider(6_001) is None


@pytest.mark.parametrize(
    "sample",
    [
        ReferencePriceSample(1_000, 999, 100.0, "binance:BTCUSDT"),
        ReferencePriceSample(1_000, 1_001, 0.0, "binance:BTCUSDT"),
        ReferencePriceSample(1_000, 1_001, float("nan"), "binance:BTCUSDT"),
        ReferencePriceSample(1_000, 1_001, 100.0, ""),
    ],
)
def test_invalid_reference_sample_is_rejected(sample: ReferencePriceSample) -> None:
    with pytest.raises(ValueError):
        sample.validate()
