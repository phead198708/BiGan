"""Contract tests for the Polymarket real corpus recorder."""

from __future__ import annotations

from pathlib import Path

import pytest

from bigan.v8.polymarket.recorder import (
    DEFAULT_SAMPLING_POLICY_SECONDS,
    PolymarketRealCorpusRecorderConfig,
)


def test_recorder_config_preserves_safe_defaults(tmp_path: Path) -> None:
    config = PolymarketRealCorpusRecorderConfig(
        run_id="safe-defaults",
        output_dir=tmp_path,
    )

    assert config.paper_only is True
    assert config.capital_at_risk is False
    assert config.broker_exchange_write_enabled is False
    assert config.live_exchange_write_enabled is False
    assert config.polymarket_write_enabled is False
    assert config.wallet_signing_enabled is False
    assert config.resolved_sampling_policy_seconds() == DEFAULT_SAMPLING_POLICY_SECONDS
    assert config.raw_dir == tmp_path.resolve() / "safe-defaults" / "raw"


@pytest.mark.parametrize(
    ("field_name", "value", "error_match"),
    (
        ("paper_only", False, "paper_only must be true"),
        ("capital_at_risk", True, "capital_at_risk must be false"),
        ("polymarket_write_enabled", True, "polymarket_write_enabled must be false"),
        ("wallet_signing_enabled", True, "wallet_signing_enabled must be false"),
    ),
)
def test_recorder_config_rejects_unsafe_flags(
    tmp_path: Path,
    field_name: str,
    value: bool,
    error_match: str,
) -> None:
    kwargs = {field_name: value}
    with pytest.raises(ValueError, match=error_match):
        PolymarketRealCorpusRecorderConfig(
            run_id="unsafe",
            output_dir=tmp_path,
            **kwargs,
        )


def test_recorder_config_rejects_invalid_sampling_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sampling intervals must be positive"):
        PolymarketRealCorpusRecorderConfig(
            run_id="bad-sampling",
            output_dir=tmp_path,
            sampling_policy_seconds={"btc_updown_5m": 0},
        )
