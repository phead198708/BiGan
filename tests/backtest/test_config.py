"""Contract tests for issue #10 backtest configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError


def _config_dict() -> dict:
    return {
        "schema_version": "backtest_config_v1",
        "strategy": {"long_threshold": 0.58},
        "costs": {"fee_bps": 2.0, "slippage_bps": 1.5},
        "execution": {"latency_ms": 750},
        "dataset": {
            "dataset_version": "features-labels-v1",
            "feature_table": "features_15m_v1",
            "label_table": "labels_15m_v1",
        },
        "model": {"model_version": "baseline-market-mid-v1"},
        "output": {"output_dir": "data/backtests"},
    }


def test_load_json_config_fills_unique_run_ids(tmp_path: Path) -> None:
    from bigan.backtest.config import load_backtest_config

    path = tmp_path / "backtest.json"
    path.write_text(json.dumps(_config_dict()), encoding="utf-8")

    first = load_backtest_config(path)
    second = load_backtest_config(path)

    assert first.strategy.long_threshold == 0.58
    assert first.costs.fee_bps == 2.0
    assert first.costs.slippage_bps == 1.5
    assert first.execution.latency_ms == 750
    assert first.dataset.dataset_version == "features-labels-v1"
    assert first.model.model_version == "baseline-market-mid-v1"
    assert re.fullmatch(r"bt-\d{8}T\d{12}Z-[0-9a-f]{8}", first.output.run_id)
    assert first.output.run_id != second.output.run_id


def test_load_yaml_config_with_fixed_schema(tmp_path: Path) -> None:
    from bigan.backtest.config import load_backtest_config

    path = tmp_path / "backtest.yaml"
    path.write_text(
        """
schema_version: backtest_config_v1
strategy:
  long_threshold: 0.61
costs:
  fee_bps: 2.5
  slippage_bps: 1.0
execution:
  latency_ms: 500
dataset:
  dataset_version: features-labels-v1
  feature_table: features_15m_v1
  label_table: labels_15m_v1
model:
  model_version: baseline-v0
output:
  output_dir: data/backtests
""".strip(),
        encoding="utf-8",
    )

    config = load_backtest_config(path)

    assert config.strategy.long_threshold == 0.61
    assert config.costs.fee_bps == 2.5
    assert config.costs.slippage_bps == 1.0
    assert config.execution.latency_ms == 500
    assert config.output.output_dir == "data/backtests"


def test_backtest_config_rejects_unknown_fields() -> None:
    from bigan.backtest.config import BacktestConfig

    payload = _config_dict()
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        BacktestConfig.model_validate(payload)


def test_backtest_config_rejects_invalid_threshold() -> None:
    from bigan.backtest.config import BacktestConfig

    payload = _config_dict()
    payload["strategy"]["long_threshold"] = 1.5

    with pytest.raises(ValidationError):
        BacktestConfig.model_validate(payload)


def test_load_config_can_preserve_supplied_run_id(tmp_path: Path) -> None:
    from bigan.backtest.config import load_backtest_config

    payload = _config_dict()
    payload["output"]["run_id"] = "bt-fixed-for-replay"
    path = tmp_path / "backtest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_backtest_config(path, new_run_id=False)

    assert config.output.run_id == "bt-fixed-for-replay"


def test_config_dumps_to_script_readable_dict() -> None:
    from bigan.backtest.config import BacktestConfig

    config = BacktestConfig.model_validate(_config_dict())
    data = config.to_script_dict()

    assert data["schema_version"] == "backtest_config_v1"
    assert data["strategy"]["long_threshold"] == 0.58
    assert data["output"]["run_id"].startswith("bt-")
    json.dumps(data)
