from __future__ import annotations

import json
from pathlib import Path

from bigan.paper_trading.operator.__main__ import _run_mock_demo, main
from bigan.paper_trading.operator.config import OperatorConfig, load_operator_config


def _config(path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            (
                'operator_id = "cli-operator"',
                'strategy_id = "cli-strategy"',
                'paper_account_id = "cli-account"',
                'source_commit = "deadbeef"',
                f'output_dir = "{output_dir}"',
                "volatility_min_samples = 1",
                "volatility_return_interval_ms = 1",
                "ofi_min_samples = 1",
                "dry_run = true",
                "mock = true",
            )
        )
        + "\n"
    )


def test_config_check_prints_sanitized_deterministic_identity(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "operator.toml"
    _config(config_path, tmp_path / "runs")
    assert main(["--config", str(config_path), "--check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["paper_only"] is True
    assert len(payload["config_sha256"]) == 64
    assert not any(word in payload for word in ("secret", "authorization", "cookie"))


async def test_local_mock_demo_uses_no_network_and_writes_final_status(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "operator.toml"
    output_dir = tmp_path / "runs"
    _config(config_path, output_dir)
    await _run_mock_demo(load_operator_config(config_path))
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "STOPPED"
    assert payload["paper_only"] is True
    assert payload["counters"]["decisions"] == 1
    assert (output_dir / "cli-operator" / "operator_status.json").is_file()
    assert len(list(output_dir.glob("paper-*"))) == 1


async def test_mock_demo_warms_up_default_twenty_samples(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("bigan.paper_trading.operator.__main__._now_ms", lambda: 500_000)
    config = OperatorConfig(
        operator_id="default-demo", strategy_id="strategy", paper_account_id="account",
        source_commit="deadbeef", output_dir=tmp_path,
    )
    await _run_mock_demo(config)
    status = json.loads(capsys.readouterr().out)
    assert status["counters"]["decisions"] >= 1
    assert status["feeds"]["binance"]["gap_count"] == 0
