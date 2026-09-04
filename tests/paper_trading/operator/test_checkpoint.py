from __future__ import annotations

from dataclasses import replace

import pytest

from bigan.paper_trading.operator.checkpoint import AccountCheckpoint, AccountCheckpointStore
from bigan.paper_trading.operator.discovery import DiscoveredMarket


def _market(index: int) -> DiscoveredMarket:
    return DiscoveredMarket(
        market_id=f"market-{index}", condition_id=f"condition-{index}",
        slug=f"btc-updown-15m-{index}", title="BTC", underlying="BTC",
        market_type="binary_up_down", window_duration_ms=900_000,
        start_ts_ms=0, end_ts_ms=900_000, yes_token_id=f"yes-{index}", no_token_id=f"no-{index}",
        active=True, closed=False, accepting_orders=True, source_endpoint="mock://market",
        discovered_at_ms=0, resolution_source="mock", resolution_identity=f"resolution-{index}",
        reference_price_at_start=100.0, raw_payload_sha256="0" * 64,
    )


def test_failed_atomic_replace_preserves_previous_account_frontier(tmp_path, monkeypatch) -> None:
    store = AccountCheckpointStore(tmp_path / "account_checkpoint.json")
    checkpoint = AccountCheckpoint(
        config_sha256="config", run_id="paper-1", market=_market(1), opening_cash=1_000,
    )
    store.write(checkpoint)

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("bigan.paper_trading.operator.checkpoint.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store.write(replace(checkpoint, run_id="paper-2", market=_market(2)))
    assert store.load(config_sha256="config") == checkpoint
    assert not list(tmp_path.glob(".account-checkpoint-*"))


def test_checkpoint_rejects_config_change_and_incomplete_cash_link(tmp_path) -> None:
    store = AccountCheckpointStore(tmp_path / "account_checkpoint.json")
    checkpoint = AccountCheckpoint(
        config_sha256="config", run_id="paper-1", market=_market(1), opening_cash=1_000,
    )
    store.write(checkpoint)
    with pytest.raises(ValueError, match="configuration identity"):
        store.load(config_sha256="different")
    with pytest.raises(ValueError, match="incomplete predecessor"):
        replace(checkpoint, predecessor_run_id="paper-old")
    with pytest.raises(ValueError, match="cash"):
        replace(checkpoint, predecessor_run_id="paper-old",
                predecessor_window_id="old-window", predecessor_settled_cash=1_100)
