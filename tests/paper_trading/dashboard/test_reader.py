from __future__ import annotations

import json
from dataclasses import replace

import pytest

from bigan.paper_trading.dashboard.reader import DashboardReader, DashboardUnavailable
from bigan.paper_trading.operator.read_model import OperatorReadRepository, OperatorState
from bigan.paper_trading.operator.runtime import PaperTradingOperator
from bigan.paper_trading.storage import IDEMPOTENCY_INDEX_FILE, MANIFEST_FILE, SNAPSHOT_FILE
from bigan.strategies.polymarket_pricing import PolymarketPricingEngine, SignalDirection
from tests.paper_trading.operator.test_runtime import (
    FakeClock,
    FakeDiscovery,
    FakeResolution,
    _config,
    _final,
    _market,
    _ready_operator,
    _selection,
)


def test_missing_status_does_not_create_output(bundle, tmp_path):
    config = replace(bundle.config, output_dir=tmp_path / "absent")
    reader = DashboardReader(config)
    with pytest.raises(DashboardUnavailable, match="Operator status is unavailable"):
        reader.read()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("state", [OperatorState.STARTING, OperatorState.DISCOVERING])
def test_legal_status_before_any_checkpoint(bundle, state):
    operator = bundle.operator
    status = replace(operator.status(), state=state, run_id=None, active_market=None)
    operator.status_writer.write(status)
    operator.checkpoint_store.path.unlink()
    view = bundle.reader.read()
    assert view["status"]["state"] == state
    assert view["account"] is None and view["positions"] is None
    assert all(value is None for value in view["recent"].values())
    assert view["warnings"][0]["code"] == "NO_ACTIVE_RUN"


@pytest.mark.parametrize("state", [OperatorState.RUNNING, OperatorState.STOPPED, OperatorState.FAILED])
def test_valid_status_states_and_canonical_values(bundle, state):
    bundle.operator.status_writer.write(replace(bundle.operator.status(), state=state))
    view = bundle.reader.read()
    expected = bundle.operator.read_repository.account_summary()
    assert view["status"]["state"] == state
    assert view["account"].items() >= expected.items()
    assert view["account"]["drawdown"] == bundle.operator.session.current_snapshot.drawdown
    assert view["account"]["run_id"] == view["status"]["run_id"]
    assert view["active_market"]["title"] == bundle.markets[0].title
    assert view["recent"]["fills"] and view["positions"]
    assert not view["warnings"]


async def test_empty_run_and_zero_capital_are_valid_observations(tmp_path, monkeypatch):
    market, clock = _market(1), FakeClock(10_000)
    config = replace(_config(tmp_path), initial_bankroll=10, max_single_trade_pct=1,
                     max_position_pct=1, max_window_exposure_pct=1, slippage_tolerance=0)
    original = PolymarketPricingEngine.evaluate_signal

    def all_in(self, **kwargs):
        return replace(original(self, **kwargs), direction=SignalDirection.BUY_YES, recommended_size_pct=1)

    monkeypatch.setattr(PolymarketPricingEngine, "evaluate_signal", all_in)
    operator = PaperTradingOperator(
        config=config, discovery=FakeDiscovery([_selection(market)]),
        resolution=FakeResolution([replace(_final(market), yes_payout=0)]), clock_ms=clock,
    )
    await operator.start()
    try:
        reader = DashboardReader(config, clock_ms=clock)
        empty = reader.read()
        assert empty["account"]["cash"] == 10
        assert empty["positions"] == []
        assert empty["recent"]["fills"] == empty["recent"]["decisions"] == []
        assert empty["warnings"] == []
        await _ready_operator(operator, clock)
        clock.now_ms = market.end_ts_ms + 1
        await operator.poll()
        exhausted = reader.read()
        assert exhausted["status"]["state"] == "EXHAUSTED"
        assert exhausted["account"]["cash"] == exhausted["account"]["equity"] == 0
        assert exhausted["account"]["realized_pnl"] == -10
        assert exhausted["account"]["drawdown"] == 1
        assert exhausted["positions"] == []
        assert exhausted["recent"]["settlements"][0]["cash_after"] == 0
        assert exhausted["warnings"] == []
    finally:
        await operator.shutdown()


@pytest.mark.parametrize("damage", [b"", b"{broken", b'{"equity":NaN}', b"null", b"[]"])
def test_corrupt_status_is_unavailable_and_sanitized(bundle, damage):
    bundle.reader.status_path.write_bytes(damage)
    with pytest.raises(DashboardUnavailable) as error:
        bundle.reader.read()
    assert str(error.value) == "Operator status is unavailable"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong_hash", "activating", "wrong_run"])
def test_checkpoint_failure_never_fabricates_account(bundle, damage):
    path = bundle.operator.checkpoint_store.path
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_text('{"bad":')
    else:
        data = json.loads(path.read_text())
        field, value = {"wrong_hash": ("config_sha256", "f" * 64), "activating": ("activation_state", "ACTIVATING"), "wrong_run": ("run_id", "paper-" + "a" * 24)}[damage]
        data[field] = value
        path.write_text(json.dumps(data))
    view = bundle.reader.read()
    assert view["status"]["run_id"] == bundle.operator.run_id
    assert view["account"] is None
    assert all(value is None for value in view["recent"].values())
    assert view["warnings"][0]["code"] == "FRONTIER_UNAVAILABLE"


@pytest.mark.parametrize("artifact", [MANIFEST_FILE, "operator_account_link.json"])
def test_missing_manifest_or_link_preserves_status_only(bundle, artifact):
    (bundle.operator.session.store.run_dir / artifact).unlink()
    view = bundle.reader.read()
    assert view["account"] is None and view["status"]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_sqlite_only_disables_fills(bundle, damage):
    path = bundle.operator.session.store.run_dir / IDEMPOTENCY_INDEX_FILE
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"not a database")
    view = bundle.reader.read()
    assert view["account"] and view["recent"]["decisions"]
    assert view["recent"]["fills"] is None
    assert view["recent"]["settlements"] == []
    assert view["warnings"] == [{"code": "HISTORY_UNAVAILABLE", "message": "Recent fills are temporarily unavailable", "section": "fills"}]
    if damage == "missing":
        assert not path.exists()


def test_one_section_error_isolated_without_exception_details(bundle, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("/private/path token=SUPER_SECRET sqlite internal failure")
    monkeypatch.setattr(OperatorReadRepository, "settlements", fail)
    view = bundle.reader.read()
    assert view["recent"]["settlements"] is None
    assert view["recent"]["fills"]
    encoded = json.dumps(view)
    assert "SUPER_SECRET" not in encoded and "/private/path" not in encoded


def test_wrong_account_run_is_unavailable_not_zero(bundle):
    path = bundle.operator.session.store.run_dir / SNAPSHOT_FILE
    data = json.loads(path.read_text())
    data["run_id"] = "paper-" + "b" * 24
    path.write_text(json.dumps(data))
    view = bundle.reader.read()
    assert view["account"] is None and view["positions"] is None
    assert view["recent"]["decisions"]


def test_stale_threshold_and_missing_pricing_values(bundle):
    bundle.clock.now_ms += bundle.reader.stale_after_ms + 1
    view = bundle.reader.read()
    assert view["stale"] is True
    assert view["stale_after_ms"] == max(5000, 3 * bundle.config.status_interval_ms)
    assert "spot_price" not in view["status"]["pricing_inputs"]


async def test_rollover_reuses_reader_but_switches_frontier_and_pages_history(bundle):
    reader, operator = bundle.reader, bundle.operator
    first = reader.read()
    bundle.clock.now_ms = bundle.markets[0].end_ts_ms + 1
    await operator.poll()
    second = reader.read()
    assert second["status"]["run_id"] != first["status"]["run_id"]
    assert second["account"]["run_id"] == second["status"]["run_id"]
    assert second["account"]["initial_bankroll"] == first["account"]["initial_bankroll"]
    assert second["account"]["cash"] == operator.session.current_snapshot.cash
    assert len(second["recent"]["runs"]) == 2
    assert second["recent"]["settlements"][0]["run_id"] == first["status"]["run_id"]
    older = reader.read(limit=1, before_run_id=second["status"]["run_id"])
    assert [r["run_id"] for r in older["recent"]["runs"]] == [first["status"]["run_id"]]
    assert "has_more" not in older
    assert reader.read(before_run_id=first["status"]["run_id"])["recent"]["runs"] == []
