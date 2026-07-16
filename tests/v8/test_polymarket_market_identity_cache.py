"""Causal Gamma market identity cache and provider fallback tests."""

from __future__ import annotations

import json
import urllib.parse
from pathlib import Path

import pytest

import examples.v8.prefetch_polymarket_gamma_market_identities as prefetch_module
from bigan.v8.polymarket import (
    GammaMarketIdentityCache,
    GammaMarketIdentityCacheError,
    PolymarketPublicHTTPRealCorpusProvider,
    PolymarketRealCorpusRecorderConfig,
    RealCorpusPublicProviderError,
)

MARKET_START_TS = 1_700_001_000_000
MARKET_END_TS = MARKET_START_TS + 300_000
MARKET_SLUG = "btc-updown-5m-1700001000"


def test_gamma_market_identity_cache_round_trip_is_causal_and_hash_verified(
    tmp_path: Path,
) -> None:
    cache = GammaMarketIdentityCache(
        tmp_path / "cache.json",
        max_age_seconds=3_600,
        current_time_ms=MARKET_START_TS,
    )
    payload = _gamma_payload(MARKET_SLUG)
    payload["outcomePrices"] = json.dumps(["0.51", "0.49"])

    stored = _store(cache, payload=payload, fetched_at_ts=MARKET_START_TS - 60_000)
    loaded = cache.lookup(
        slug=MARKET_SLUG,
        decision_ts=MARKET_START_TS + 1_000,
        expected_market_family="btc_updown_5m",
        expected_market_start_ts=MARKET_START_TS,
        expected_market_end_ts=MARKET_END_TS,
    )

    assert loaded["cache_entry_sha256"] == stored["cache_entry_sha256"]
    assert loaded["cache_age_ms"] == 61_000
    assert loaded["cache_provenance_valid"] is True
    assert loaded["paper_only"] is True
    assert loaded["capital_at_risk"] is False
    assert "outcomePrices" in loaded["raw_public_payload"]
    assert "outcomePrices" not in loaded["identity_payload"]
    assert loaded["forbidden_fields_used_for_identity"] == []


def test_gamma_market_identity_cache_rejects_post_start_prefetch(
    tmp_path: Path,
) -> None:
    cache = GammaMarketIdentityCache(
        tmp_path / "cache.json",
        max_age_seconds=3_600,
    )

    with pytest.raises(GammaMarketIdentityCacheError) as error:
        _store(
            cache,
            payload=_gamma_payload(MARKET_SLUG),
            fetched_at_ts=MARKET_START_TS + 1,
        )

    assert error.value.reason_codes == (
        "gamma_market_identity_cache_post_start_prefetch",
    )


def test_gamma_market_identity_cache_rejects_payload_identity_mismatch(
    tmp_path: Path,
) -> None:
    cache = GammaMarketIdentityCache(
        tmp_path / "cache.json",
        max_age_seconds=3_600,
    )
    payload = _gamma_payload(MARKET_SLUG)

    with pytest.raises(GammaMarketIdentityCacheError) as error:
        cache.store_prefetched_payload(
            payload=payload,
            slug=MARKET_SLUG,
            market_family="btc_updown_5m",
            market_start_ts=MARKET_START_TS,
            market_end_ts=MARKET_END_TS,
            condition_id="wrong-condition",
            up_token_id="wrong-up",
            down_token_id="wrong-down",
            reference_price_source=(
                "https://data.chain.link/streams/btc-usd"
            ),
            settlement_rule="UP if the official end reference is above start.",
            fetched_at_ts=MARKET_START_TS - 60_000,
            source_endpoint="https://gamma-api.polymarket.com/markets",
        )

    assert error.value.reason_codes == (
        "gamma_market_identity_cache_payload_identity_mismatch",
    )


def test_gamma_market_identity_cache_rejects_stale_entry(tmp_path: Path) -> None:
    cache = GammaMarketIdentityCache(
        tmp_path / "cache.json",
        max_age_seconds=30,
    )
    _store(
        cache,
        payload=_gamma_payload(MARKET_SLUG),
        fetched_at_ts=MARKET_START_TS - 60_000,
    )

    with pytest.raises(GammaMarketIdentityCacheError) as error:
        cache.lookup(
            slug=MARKET_SLUG,
            decision_ts=MARKET_START_TS + 1_000,
            expected_market_family="btc_updown_5m",
            expected_market_start_ts=MARKET_START_TS,
            expected_market_end_ts=MARKET_END_TS,
        )

    assert error.value.reason_codes == ("gamma_market_identity_cache_stale",)


def test_gamma_market_identity_cache_rejects_tampered_entry_hash(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    cache = GammaMarketIdentityCache(cache_path, max_age_seconds=3_600)
    _store(
        cache,
        payload=_gamma_payload(MARKET_SLUG),
        fetched_at_ts=MARKET_START_TS - 60_000,
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["entries"][MARKET_SLUG]["up_token_id"] = "tampered-token"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GammaMarketIdentityCacheError) as error:
        cache.lookup(
            slug=MARKET_SLUG,
            decision_ts=MARKET_START_TS + 1_000,
            expected_market_family="btc_updown_5m",
            expected_market_start_ts=MARKET_START_TS,
            expected_market_end_ts=MARKET_END_TS,
        )

    assert error.value.reason_codes == (
        "gamma_market_identity_cache_hash_mismatch",
    )


def test_provider_uses_cache_only_after_gamma_timeout_and_clob_revalidation(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    cache = GammaMarketIdentityCache(cache_path, max_age_seconds=3_600)
    _store(
        cache,
        payload=_gamma_payload(MARKET_SLUG),
        fetched_at_ts=MARKET_START_TS - 60_000,
    )
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=MARKET_START_TS + 1_000,
        fetch_json=GammaFailureClobFetch(),
        market_identity_cache_path=cache_path,
        market_identity_cache_max_age_seconds=3_600,
    )

    rows = provider.market_rows(_config(tmp_path))

    assert len(rows) == 1
    assert rows[0]["market_identity_source_type"] == (
        "gamma_prefetch_cache_fallback"
    )
    assert rows[0]["market_identity_cache_fallback_used"] is True
    assert rows[0]["market_identity_cache_fallback_reason_codes"] == [
        "read_only_public_http_timeout"
    ]
    assert rows[0]["market_identity_cache_provenance_valid"] is True
    assert rows[0]["market_identity_clob_revalidation_passed"] is True
    assert rows[0]["market_identity_live_orderbook_validation_required"] is True
    assert rows[0]["paper_only"] is True
    assert rows[0]["capital_at_risk"] is False


def test_provider_retries_transient_clob_identity_timeout_without_relaxing_identity(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    cache = GammaMarketIdentityCache(cache_path, max_age_seconds=3_600)
    _store(
        cache,
        payload=_gamma_payload(MARKET_SLUG),
        fetched_at_ts=MARKET_START_TS - 60_000,
    )
    fetch = GammaFailureClobFetch(transient_clob_timeout_count=1)
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=MARKET_START_TS + 1_000,
        fetch_json=fetch,
        market_identity_cache_path=cache_path,
        market_identity_cache_max_age_seconds=3_600,
        clob_identity_revalidation_max_attempts=3,
        clob_identity_revalidation_retry_seconds=0.0,
    )

    rows = provider.market_rows(_config(tmp_path))

    assert fetch.clob_calls == 2
    revalidation = rows[0]["market_identity_clob_revalidation"]
    assert revalidation["attempt_count"] == 2
    assert revalidation["retry_reason_codes"] == [
        "read_only_public_http_timeout"
    ]
    assert revalidation["retry_policy_relaxed_identity_checks"] is False
    assert rows[0]["market_identity_clob_revalidation_passed"] is True


def test_provider_fails_closed_when_gamma_times_out_and_cache_is_missing(
    tmp_path: Path,
) -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=MARKET_START_TS + 1_000,
        fetch_json=GammaFailureClobFetch(),
        market_identity_cache_path=tmp_path / "cache.json",
        market_identity_cache_max_age_seconds=3_600,
    )

    with pytest.raises(RealCorpusPublicProviderError) as error:
        provider.market_rows(_config(tmp_path))

    assert "read_only_public_http_timeout" in error.value.reason_codes
    assert "gamma_market_identity_cache_missing" in error.value.reason_codes


def test_provider_fails_closed_when_cached_token_mapping_disagrees_with_clob(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    cache = GammaMarketIdentityCache(cache_path, max_age_seconds=3_600)
    _store(
        cache,
        payload=_gamma_payload(MARKET_SLUG),
        fetched_at_ts=MARKET_START_TS - 60_000,
    )
    fetch = GammaFailureClobFetch(down_token_id="wrong-down-token")
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=MARKET_START_TS + 1_000,
        fetch_json=fetch,
        market_identity_cache_path=cache_path,
        market_identity_cache_max_age_seconds=3_600,
        clob_identity_revalidation_retry_seconds=0.0,
    )

    with pytest.raises(RealCorpusPublicProviderError) as error:
        provider.market_rows(_config(tmp_path))

    assert error.value.reason_codes == (
        "gamma_market_identity_cache_clob_token_mismatch",
    )
    assert fetch.clob_calls == 1


def test_provider_prefetches_deterministic_future_slugs_before_market_start(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    provider = PolymarketPublicHTTPRealCorpusProvider(
        current_time_ms=MARKET_START_TS + 1_000,
        fetch_json=FutureGammaFetch(),
        market_identity_cache_path=cache_path,
        market_identity_cache_max_age_seconds=7_200,
        gamma_market_identity_prefetch_round_count=2,
    )

    report = provider.prefetch_gamma_market_identities(
        config=_config(tmp_path),
        base_slugs=(MARKET_SLUG,),
    )

    assert report["stored_slugs"] == [
        "btc-updown-5m-1700001300",
        "btc-updown-5m-1700001600",
    ]
    assert report["reason_codes"] == []
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(payload["entries"]) == set(report["stored_slugs"])
    assert all(
        int(row["fetched_at_ts"]) <= int(row["market_start_ts"])
        for row in payload["entries"].values()
    )


def test_prefetch_runner_writes_hashable_fail_closed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def prefetch_gamma_market_identities(self, *, config):
            assert config.market_families == ("btc_updown_5m",)
            return {
                "prefetch_enabled": True,
                "requested_slug_count": 2,
                "stored_slug_count": 2,
                "stored_slugs": ["future-a", "future-b"],
                "reason_codes": [],
                "cache_report": {
                    "cache_payload_sha256": "b" * 64,
                },
                "paper_only": True,
                "capital_at_risk": False,
            }

    monkeypatch.setattr(
        prefetch_module,
        "PolymarketPublicHTTPRealCorpusProvider",
        FakeProvider,
    )
    result = prefetch_module.run_gamma_market_identity_prefetch_cli(
        run_id="prefetch",
        output_dir=tmp_path,
        cache_path=tmp_path / "cache.json",
        market_family="btc_updown_5m",
        prefetch_round_count=2,
        http_timeout_seconds=5.0,
        cache_max_age_seconds=7_200.0,
    )

    assert result["report"]["stored_slug_count"] == 2
    assert result["report"]["outcome_or_pnl_fields_used"] is False
    assert result["manifest"]["cache_payload_sha256"] == "b" * 64
    assert len(result["report_sha256"]) == 64
    assert len(result["manifest_sha256"]) == 64
    assert result["manifest"]["paper_only"] is True
    assert result["manifest"]["capital_at_risk"] is False


def _config(tmp_path: Path) -> PolymarketRealCorpusRecorderConfig:
    return PolymarketRealCorpusRecorderConfig(
        run_id="market-identity-cache",
        output_dir=tmp_path,
        market_families=("btc_updown_5m",),
        mock_public_data=False,
    )


def _store(
    cache: GammaMarketIdentityCache,
    *,
    payload: dict,
    fetched_at_ts: int,
) -> dict:
    condition_id = str(payload["conditionId"])
    up_token_id, down_token_id = json.loads(payload["clobTokenIds"])
    return cache.store_prefetched_payload(
        payload=payload,
        slug=MARKET_SLUG,
        market_family="btc_updown_5m",
        market_start_ts=MARKET_START_TS,
        market_end_ts=MARKET_END_TS,
        condition_id=condition_id,
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        reference_price_source="https://data.chain.link/streams/btc-usd",
        settlement_rule="UP if the official end reference is above start.",
        fetched_at_ts=fetched_at_ts,
        source_endpoint="https://gamma-api.polymarket.com/markets",
    )


def _gamma_payload(slug: str) -> dict:
    start_seconds = int(slug.rsplit("-", 1)[-1])
    return {
        "conditionId": f"0xcondition-{start_seconds}",
        "slug": slug,
        "question": "Bitcoin Up or Down - test",
        "description": "UP if the official end reference is above start.",
        "resolutionSource": "https://data.chain.link/streams/btc-usd",
        "priceToBeat": "65000",
        "outcomes": json.dumps(["Up", "Down"]),
        "clobTokenIds": json.dumps(
            [f"up-token-{start_seconds}", f"down-token-{start_seconds}"]
        ),
    }


class GammaFailureClobFetch:
    def __init__(
        self,
        *,
        down_token_id: str | None = None,
        transient_clob_timeout_count: int = 0,
    ) -> None:
        self.down_token_id = down_token_id
        self.transient_clob_timeout_count = transient_clob_timeout_count
        self.clob_calls = 0

    def __call__(self, url: str):
        parsed = urllib.parse.urlparse(url)
        if "gamma-api.polymarket.com" in parsed.netloc:
            raise RealCorpusPublicProviderError(
                "Gamma timed out.",
                reason_codes=("read_only_public_http_timeout",),
            )
        if "clob.polymarket.com" in parsed.netloc:
            self.clob_calls += 1
            if self.clob_calls <= self.transient_clob_timeout_count:
                raise RealCorpusPublicProviderError(
                    "CLOB timed out.",
                    reason_codes=("read_only_public_http_timeout",),
                )
            payload = _gamma_payload(MARKET_SLUG)
            up_token_id, expected_down_token_id = json.loads(
                payload["clobTokenIds"]
            )
            return {
                "condition_id": payload["conditionId"],
                "market_slug": MARKET_SLUG,
                "tokens": [
                    {"token_id": up_token_id, "outcome": "Up"},
                    {
                        "token_id": (
                            self.down_token_id or expected_down_token_id
                        ),
                        "outcome": "Down",
                    },
                ],
            }
        raise AssertionError(f"unexpected url: {url}")


class FutureGammaFetch:
    def __call__(self, url: str):
        parsed = urllib.parse.urlparse(url)
        if "gamma-api.polymarket.com" not in parsed.netloc:
            raise AssertionError(f"unexpected url: {url}")
        slug = urllib.parse.parse_qs(parsed.query)["slug"][0]
        return [_gamma_payload(slug)]
