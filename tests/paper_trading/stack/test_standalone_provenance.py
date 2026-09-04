from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bigan import build_provenance
from bigan.paper_trading.dashboard import __main__ as dashboard_cli
from bigan.paper_trading.operator import __main__ as operator_cli
from tests.paper_trading.stack.conftest import COMMIT, write_config


@pytest.fixture(params=["operator", "dashboard"])
def entrypoint(request, monkeypatch):
    # The actual CLI validation runs, but no network/listener/writer may start.
    if request.param == "operator":
        start = AsyncMock()
        monkeypatch.setattr(operator_cli, "_run_live", start)
        return SimpleNamespace(main=operator_cli.main, start=start)
    start = Mock()
    monkeypatch.setattr(dashboard_cli.web, "run_app", start)
    monkeypatch.setattr(dashboard_cli, "DashboardReader", Mock())
    monkeypatch.setattr(dashboard_cli, "create_app", Mock())
    return SimpleNamespace(main=dashboard_cli.main, start=start)


@pytest.mark.parametrize("damage", ["unsealed", "altered", "mismatched"])
def test_standalone_live_rejects_unverified_identity_before_start(entrypoint, tmp_path, monkeypatch, damage):
    path = write_config(tmp_path, source_commit="0" * 40)

    def provenance():
        if damage != "mismatched":
            code = "BUILD_PROVENANCE_UNAVAILABLE" if damage == "unsealed" else "ALTERED_BUILD_CONTENTS"
            raise build_provenance.BuildProvenanceError(code)
        return {"source_commit": COMMIT}

    monkeypatch.setattr(build_provenance, "runtime_provenance", provenance)
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["--config", str(path)])  # No hidden expected-source argument.
    assert exc.value.code == 2
    entrypoint.start.assert_not_called()
    assert not (tmp_path / "paper-output").exists()


def test_standalone_live_accepts_matching_verified_identity(entrypoint, tmp_path, monkeypatch):
    path = write_config(tmp_path)
    verifier = Mock(return_value={"source_commit": COMMIT})
    monkeypatch.setattr(build_provenance, "runtime_provenance", verifier)
    assert entrypoint.main(["--config", str(path)]) == 0
    verifier.assert_called_once_with()
    entrypoint.start.assert_called_once()
    assert not (tmp_path / "paper-output").exists()


def test_expected_identity_cannot_override_config_identity(entrypoint, tmp_path, monkeypatch):
    path = write_config(tmp_path, source_commit="0" * 40)
    monkeypatch.setattr(build_provenance, "runtime_provenance", lambda: {"source_commit": COMMIT})
    with pytest.raises(SystemExit) as exc:
        entrypoint.main(["--config", str(path), "--expected-source-commit", COMMIT])
    assert exc.value.code == 2
    entrypoint.start.assert_not_called()
    assert not (tmp_path / "paper-output").exists()


@pytest.mark.parametrize("mode", ["check", "config_check_only", "dry_run", "mock", "mock_demo"])
def test_operator_nonlive_modes_do_not_require_a_sealed_build(tmp_path, monkeypatch, mode):
    overrides = {mode: True} if mode in {"config_check_only", "dry_run", "mock"} else {}
    path = write_config(tmp_path, **overrides)
    live, mock = AsyncMock(), AsyncMock()
    monkeypatch.setattr(operator_cli, "_run_live", live)
    monkeypatch.setattr(operator_cli, "_run_mock_demo", mock)
    monkeypatch.setattr(build_provenance, "runtime_provenance", Mock(side_effect=AssertionError("not live")))
    args = ["--config", str(path)] + ({"check": ["--check"], "mock_demo": ["--mock-demo"]}.get(mode, []))
    assert operator_cli.main(args) == 0
    live.assert_not_called()
    assert mock.call_count == (1 if mode in {"mock", "mock_demo"} else 0)
    assert not (tmp_path / "paper-output").exists()
