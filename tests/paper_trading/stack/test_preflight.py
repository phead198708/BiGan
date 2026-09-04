from __future__ import annotations

import socket

import pytest

from bigan.paper_trading.stack.__main__ import main
from bigan.paper_trading.stack.preflight import SAFETY, PreflightError, duration_seconds, preflight
from tests.paper_trading.stack.conftest import free_port, write_config


def test_valid_live_is_pure(tmp_path, monkeypatch):
    path = write_config(tmp_path, output_dir="relative-paper")
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.rglob("*"))
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("preflight connected"))
    result = preflight(config_path=path, port=free_port(), report_dir=tmp_path / "reports")
    assert result.config.config_sha256 == result.identity["config_sha256"]
    assert str(result.config.output_dir) == "relative-paper"
    assert result.cwd == tmp_path
    assert result.summary()["mode"] == "live_public_feeds_paper_execution"
    assert list(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("key", list(SAFETY))
def test_all_safety_fields_are_exact(tmp_path, key):
    path = write_config(tmp_path, **{key: not SAFETY[key]})
    with pytest.raises(PreflightError):
        preflight(config_path=path, port=free_port())
    assert not (tmp_path / "paper-output").exists()


@pytest.mark.parametrize("key,value", [
    ("mock", True), ("dry_run", True), ("config_check_only", True),
    ("source_commit", "unknown"), ("source_commit", "replace-with-deployed-git-commit"),
    ("source_commit", ""), ("source_commit", "a" * 40), ("source_commit", "deadbeef" * 5),
    ("source_commit", "0123456789" * 4), ("source_commit", "examples"),
    ("wallet", "private-value"), ("api_key", "secret-example"), ("my_secret", "private-value"),
    ("gamma_markets_endpoint", "https://private.invalid/private"),
])
def test_invalid_configuration_is_sanitized(tmp_path, key, value):
    path = write_config(tmp_path, **{key: value})
    with pytest.raises(PreflightError) as error:
        preflight(config_path=path, port=free_port())
    assert str(tmp_path) not in str(error.value)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("host,port", [("0.0.0.0", 8080), ("::", 8080), ("localhost", 8080),
                                        ("example.com", 8080), ("127.0.0.1", 0), ("127.0.0.1", 65536)])
def test_listener_validation(tmp_path, host, port):
    with pytest.raises(PreflightError):
        preflight(config_path=write_config(tmp_path), host=host, port=port)


def test_occupied_port(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(PreflightError, match="PORT_UNAVAILABLE"):
            preflight(config_path=write_config(tmp_path), port=sock.getsockname()[1])


@pytest.mark.parametrize("kind", ["inside", "ancestor", "nonempty", "symlink"])
def test_report_boundaries(tmp_path, kind):
    path = write_config(tmp_path)
    report = tmp_path / "paper-output" / "report" if kind == "inside" else tmp_path / "report"
    if kind == "ancestor":
        report = tmp_path
    if kind == "nonempty":
        report.mkdir()
        (report / "soak_report.json").write_text("existing")
    if kind == "symlink":
        report.symlink_to(tmp_path / "paper-output", target_is_directory=True)
    with pytest.raises(PreflightError):
        preflight(config_path=path, report_dir=report, port=free_port())


@pytest.mark.parametrize("value", ["0s", "-1s", "1.5s", "nan", "infs", "1e3s", "30", " 3s", "8d", "9999999h", "1681h"])
def test_duration_invalid(value):
    with pytest.raises(ValueError):
        duration_seconds(value)


@pytest.mark.parametrize("value,expected", [("30s", 30), ("15m", 900), ("2h", 7200)])
def test_duration_valid(value, expected):
    assert duration_seconds(value) == expected


def test_cli_preflight_safe_summary(tmp_path, capsys):
    assert main(["--config", str(write_config(tmp_path)), "--preflight", "--dashboard-port", str(free_port())]) == 0
    output = capsys.readouterr().out
    assert "config_sha256" in output and "output_dir" not in output and str(tmp_path) not in output


def test_cli_no_implicit_report_location(tmp_path):
    with pytest.raises(SystemExit):
        main(["--config", str(write_config(tmp_path))])
