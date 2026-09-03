"""Paper operator configuration and safety-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bigan.paper_trading.operator.config import (
    OperatorConfig,
    load_operator_config,
    operator_config_from_mapping,
)


def _minimal(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "operator_id": "btc-paper-operator",
        "strategy_id": "alpha-pricing-v1",
        "paper_account_id": "paper-account-a",
        "source_commit": "816e88f",
        "output_dir": str(tmp_path),
    }
    values.update(overrides)
    return values


def test_defaults_are_strictly_paper_only_and_identity_is_stable(tmp_path: Path) -> None:
    first = OperatorConfig(**_minimal(tmp_path))
    second = OperatorConfig(**_minimal(tmp_path))

    assert first.paper_only is True
    assert first.capital_at_risk is False
    assert first.broker_exchange_write_enabled is False
    assert first.live_exchange_write_enabled is False
    assert first.polymarket_write_enabled is False
    assert first.wallet_signing_enabled is False
    assert first.config_identity() == second.config_identity()
    assert first.config_sha256 == second.config_sha256
    assert len(first.config_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("paper_only", False),
        ("capital_at_risk", True),
        ("broker_exchange_write_enabled", True),
        ("live_exchange_write_enabled", True),
        ("polymarket_write_enabled", True),
        ("wallet_signing_enabled", True),
        ("live_trading", True),
        ("private_key", "secret"),
        ("api_token", "secret"),
        ("authorization", "Bearer secret"),
    ],
)
def test_dangerous_configuration_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _minimal(tmp_path, **{field: value})
    with pytest.raises(ValueError, match="safety|dangerous|unknown"):
        operator_config_from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "endpoint"),
    [
        ("gamma_markets_endpoint", "https://gamma-api.polymarket.com/orders"),
        ("polymarket_ws_url", "wss://clob.polymarket.com/ws/create-order"),
        ("binance_depth_endpoint", "https://api.binance.com/api/v3/order"),
        ("binance_ws_url", "wss://user:password@stream.binance.com/ws"),
        ("resolution_endpoint", "https://gamma-api.polymarket.com/cancel-order"),
    ],
)
def test_write_or_authenticated_endpoints_are_rejected(
    tmp_path: Path,
    field: str,
    endpoint: str,
) -> None:
    with pytest.raises(ValueError, match="read-only endpoint"):
        OperatorConfig(**_minimal(tmp_path, **{field: endpoint}))


def test_ranges_and_queue_bounds_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="queue_size"):
        OperatorConfig(**_minimal(tmp_path, market_queue_size=0))
    with pytest.raises(ValueError, match="reconnect"):
        OperatorConfig(
            **_minimal(
                tmp_path,
                reconnect_min_seconds=10.0,
                reconnect_max_seconds=1.0,
            )
        )
    with pytest.raises(ValueError, match="risk"):
        OperatorConfig(**_minimal(tmp_path, max_position_pct=1.1))
    with pytest.raises(ValueError, match="only 5m and 15m"):
        OperatorConfig(**_minimal(tmp_path, window_duration_ms=3_600_000))
    with pytest.raises(ValueError, match="binance_book_level_limit"):
        OperatorConfig(**_minimal(tmp_path, binance_book_level_limit=0))


def test_toml_loader_rejects_unknown_fields_and_loads_safe_file(tmp_path: Path) -> None:
    path = tmp_path / "operator.toml"
    path.write_text(
        """
operator_id = "btc-paper-operator"
strategy_id = "alpha-pricing-v1"
paper_account_id = "paper-account-a"
source_commit = "816e88f"
output_dir = "./paper-runs"
mock = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_operator_config(path)
    assert config.operator_id == "btc-paper-operator"
    assert config.output_dir == Path("./paper-runs")
    assert config.mock is True

    path.write_text(path.read_text() + "private_key = \"bad\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dangerous"):
        load_operator_config(path)
