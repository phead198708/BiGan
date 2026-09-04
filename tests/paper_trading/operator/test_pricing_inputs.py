"""Dynamic spot/oracle/TWAP/volatility provider tests."""

from __future__ import annotations

import math

import pytest

from bigan.paper_trading.operator.pricing_inputs import (
    ReferencePriceSample,
    RollingPricingInputsProvider,
)


def _provider(*, max_samples: int = 10, min_returns: int = 2, lookback: int | None = None) -> RollingPricingInputsProvider:
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
        oracle_twap_lookback_seconds=lookback,
    )


def _sample(ts_ms: int, price: float, source: str) -> ReferencePriceSample:
    return ReferencePriceSample(
        timestamp_ms=ts_ms,
        received_at_ms=ts_ms + 1,
        price=price,
        source=source,
    )


@pytest.mark.parametrize("lookback", [30, 60])
def test_published_twap_is_not_window_average_and_uses_its_own_volatility(lookback):
    provider = _provider(lookback=lookback)
    for timestamp, spot, twap in [(1000, 100, 110), (2000, 200, 111), (3000, 50, 109)]:
        assert provider.ingest_spot(_sample(timestamp, spot, provider.spot_source))
        assert provider.ingest_oracle(_sample(timestamp, twap, provider.oracle_source))
    inputs = provider(3000)
    assert inputs.spot_price == 50  # Binance remains an independent observed input.
    assert inputs.oracle_twap_so_far == 109  # Latest published TWAP, no second averaging.
    assert inputs.twap_weight == 0
    returns = [math.log(111 / 110), math.log(109 / 111)]
    mean = sum(returns) / 2
    expected = math.sqrt(sum((r - mean) ** 2 for r in returns) / 2 * 31536000)
    assert inputs.volatility_annualized == pytest.approx(expected)
    assert provider.config_identity()["volatility_source"] == "published_twap"
    assert provider(4001) is None  # Independent source freshness is still mandatory.
    provider.reset_for_reconnect()
    assert provider(3000) is None


def test_spot_updates_cannot_warm_published_twap_model():
    provider = _provider(lookback=60)
    for timestamp in (1000, 2000, 3000):
        provider.ingest_spot(_sample(timestamp, 100, provider.spot_source))
    provider.ingest_oracle(_sample(3000, 100, provider.oracle_source))
    assert provider.return_sample_count == 0
    assert provider(3000) is None


def test_published_twap_outlier_is_atomic_and_does_not_advance_time():
    provider = _provider(lookback=60)
    provider.ingest_oracle(_sample(1000, 100, provider.oracle_source))
    assert not provider.ingest_oracle(_sample(2000, 200, provider.oracle_source))
    assert provider.last_oracle_timestamp_ms == 1000
    assert provider.return_sample_count == 0


def test_long_outage_return_is_not_assigned_to_short_volatility_window() -> None:
    provider = _provider(min_returns=1)
    assert provider.ingest_spot(_sample(1_000, 100, provider.spot_source))
    assert provider.ingest_spot(_sample(2_000, 101, provider.spot_source))
    assert provider.return_sample_count == 1
    assert provider.ingest_spot(_sample(3_600_000, 200, provider.spot_source))
    assert provider.return_sample_count == 0
    provider.ingest_oracle(_sample(3_600_000, 200, provider.oracle_source))
    assert provider.health(now_ms=3_600_000).ready is False
    assert provider(3_600_000) is None
    assert provider.ingest_spot(_sample(3_601_000, 201, provider.spot_source))
    assert provider.return_sample_count == 1


def test_return_eviction_uses_start_time_and_health_expires_warmup() -> None:
    provider = _provider(min_returns=1)
    provider.ingest_spot(_sample(0, 100, provider.spot_source))
    provider.ingest_spot(_sample(10_000, 101, provider.spot_source))
    provider.ingest_oracle(_sample(10_000, 100, provider.oracle_source))
    assert provider.health(now_ms=10_000).ready is True
    assert provider.health(now_ms=10_001).ready is False
    assert provider.return_sample_count == 0


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


def test_oracle_twap_is_window_scoped_and_time_weighted() -> None:
    provider = RollingPricingInputsProvider(
        window_start_ts_ms=1_000,
        window_end_ts_ms=11_000,
        spot_source="binance:BTCUSDT",
        oracle_source="polymarket-rtds:btc/usd",
        max_age_ms=1_000,
        max_samples=10,
        twap_window_ms=10_000,
        return_interval_ms=1_000,
        volatility_window_ms=10_000,
        volatility_min_samples=1,
        volatility_max_abs_log_return=0.20,
        annualization_seconds=31_536_000,
    )
    provider.ingest_spot(_sample(1_000, 100.0, "binance:BTCUSDT"))
    provider.ingest_spot(_sample(2_000, 101.0, "binance:BTCUSDT"))
    provider.ingest_oracle(_sample(900, 10.0, "polymarket-rtds:btc/usd"))
    provider.ingest_oracle(_sample(1_100, 20.0, "polymarket-rtds:btc/usd"))
    provider.ingest_oracle(_sample(2_000, 30.0, "polymarket-rtds:btc/usd"))

    inputs = provider(2_000)

    assert inputs is not None
    # The pre-open sample is carried from the 1,000 window boundary, not
    # integrated before it. The event at decision time has zero duration.
    assert inputs.oracle_twap_so_far == pytest.approx(19.0)
    assert provider.config_identity()["twap_sampling"] == "event_time_left_continuous"


def test_volatility_annualization_uses_actual_return_intervals() -> None:
    def measured_volatility(spacing_ms: int) -> float:
        provider = RollingPricingInputsProvider(
            window_start_ts_ms=0,
            window_end_ts_ms=spacing_ms * 3,
            spot_source="binance:BTCUSDT",
            oracle_source="polymarket-rtds:btc/usd",
            max_age_ms=spacing_ms,
            max_samples=10,
            twap_window_ms=spacing_ms * 3,
            return_interval_ms=1_000,
            volatility_window_ms=spacing_ms * 3,
            volatility_min_samples=2,
            volatility_max_abs_log_return=0.20,
            annualization_seconds=31_536_000,
        )
        provider.ingest_spot(_sample(0, 100.0, "binance:BTCUSDT"))
        provider.ingest_spot(
            _sample(spacing_ms, 100.0 * math.exp(0.01), "binance:BTCUSDT")
        )
        provider.ingest_spot(_sample(spacing_ms * 2, 100.0, "binance:BTCUSDT"))
        provider.ingest_oracle(_sample(0, 100.0, "polymarket-rtds:btc/usd"))
        provider.ingest_oracle(
            _sample(spacing_ms * 2, 100.0, "polymarket-rtds:btc/usd")
        )
        inputs = provider(spacing_ms * 2)
        assert inputs is not None
        return inputs.volatility_annualized

    one_second = measured_volatility(1_000)
    ten_seconds = measured_volatility(10_000)

    assert ten_seconds == pytest.approx(one_second / math.sqrt(10.0))


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
