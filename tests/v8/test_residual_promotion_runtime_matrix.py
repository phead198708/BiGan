from __future__ import annotations

import importlib.util
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/v8/verify_residual_promotion_runtime_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("runtime_matrix_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_runtime_matrix_matches_deployment_environment() -> None:
    _module().main()


def test_runtime_matrix_version_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    matrix = json.loads(module.MATRIX_PATH.read_bytes())
    matrix["xgboost_version"] = "3.4.0"
    path = tmp_path / "runtime_matrix.json"
    path.write_text(json.dumps(matrix, sort_keys=True) + "\n")
    monkeypatch.setattr(module, "MATRIX_PATH", path)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module.main()


def test_installed_xgboost_version_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module.xgb, "__version__", "3.2.1")
    with pytest.raises(RuntimeError, match="deployment runtime matrix"):
        module.main()


def test_runtime_lock_byte_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    runtime_lock = json.loads(module.LOCK_PATH.read_bytes())
    runtime_lock["packages"]["numpy"] = "2.4.7"
    path = tmp_path / "runtime.lock.json"
    path.write_text(json.dumps(runtime_lock, sort_keys=True) + "\n")
    monkeypatch.setattr(module, "LOCK_PATH", path)
    with pytest.raises(RuntimeError, match="runtime lock SHA-256 mismatch"):
        module.main()


def test_project_and_ci_use_the_frozen_runtime_matrix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == "==3.12.4"
    assert {"numpy==2.4.6", "scipy==1.17.1", "xgboost==3.2.0"} <= set(
        project["dependencies"]
    )
    workflow = (ROOT / ".github/workflows/v8-phase0.yml").read_text()
    assert 'python-version-file: ".python-version"' in workflow
    assert "verify_residual_promotion_runtime_matrix.py" in workflow
