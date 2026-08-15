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


def test_frozen_runtime_matrix_static_contract_is_exact() -> None:
    _module().verify_static_contract()


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
        module.verify_static_contract()


def test_installed_xgboost_version_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module.xgb, "__version__", "3.2.1")
    with pytest.raises(RuntimeError, match="deployment runtime matrix"):
        matrix, _ = module.verify_static_contract()
        module._verify_model_runtime_versions(matrix)


def test_runtime_lock_byte_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    runtime_lock = json.loads(module.LOCK_PATH.read_bytes())
    runtime_lock["requirements"]["package_count"] = 51
    path = tmp_path / "runtime.lock.json"
    path.write_text(json.dumps(runtime_lock, sort_keys=True) + "\n")
    monkeypatch.setattr(module, "LOCK_PATH", path)
    with pytest.raises(RuntimeError, match="runtime lock SHA-256 mismatch"):
        module.verify_static_contract()


def test_requirements_lock_byte_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    path = tmp_path / "requirements.lock.txt"
    path.write_bytes(module.REQUIREMENTS_LOCK_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(module, "REQUIREMENTS_LOCK_PATH", path)
    with pytest.raises(RuntimeError, match="requirements lock SHA-256 mismatch"):
        module.verify_static_contract()


def test_project_and_ci_use_the_frozen_runtime_matrix() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == "==3.12.4"
    assert {"numpy==2.4.6", "scipy==1.17.1", "xgboost==3.2.0"} <= set(
        project["dependencies"]
    )
    workflow = (ROOT / ".github/workflows/v8-phase0.yml").read_text()
    assert (
        "python:3.12.4-slim-bookworm@"
        "sha256:a074fac67aa01841fee592d00bae14d25dcaf98ef6e12a683ecceb7e0147e2d1"
        in workflow
    )
    assert "--require-hashes --only-binary=:all:" in workflow
    assert "--no-deps --no-build-isolation -e ." in workflow
    assert 'python-version-file: ".python-version"' in workflow
    assert "verify_residual_promotion_runtime_matrix.py --static-only" in workflow
    assert "verify_residual_promotion_runtime_matrix.py" in workflow
    assert 'PYTHONPATH: "src:."' in workflow
    assert workflow.index(
        "Verify exact deployment image, ABI, and distribution graph"
    ) < workflow.index(
        "Install additive execution-gateway runtime graph",
        workflow.index("exact-deployment-runtime:"),
    )
    assert "python -m pip check" in workflow
    assert "BIGAN_DEPLOYMENT_IMAGE_MANIFEST_DIGEST" not in workflow
    assert "tests/v8/test_residual_promotion_micro_live_executor.py" in workflow
    assert (
        "tests/v8/test_residual_promotion_v1.py::"
        "test_repository_local_bundle_loads_and_matches_frozen_parity"
    ) in workflow
