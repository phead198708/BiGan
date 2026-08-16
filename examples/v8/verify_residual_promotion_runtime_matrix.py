#!/usr/bin/env python3
"""Fail closed unless the frozen promotion deployment runtime is exact."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import sysconfig
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import xgboost as xgb

from bigan.v8.polymarket import residual_promotion_micro_live_executor as executor

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "examples/v8/residual_promotion_runtime_matrix.json"
LOCK_PATH = ROOT / "examples/v8/residual_promotion_runtime.lock.json"
REQUIREMENTS_LOCK_PATH = ROOT / (
    "examples/v8/residual_promotion_runtime-linux-x86_64.lock.txt"
)
WORKFLOW_PATH = ROOT / ".github/workflows/v8-phase0.yml"
PYTHON_VERSION_PATH = ROOT / ".python-version"
CANDIDATE_MANIFEST_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1/"
    "candidate_bundle/bundle_manifest.json"
)
EXPECTED_MATRIX_SHA256 = (
    "7a200092fc90cdecc358114fbd22972f575d07ef0a0a74ece0ece0ea46e2a42f"
)
EXPECTED_LOCK_SHA256 = (
    "74036bc0c40ca2552e44074310f85ecf807bb51557ffad99eb815f97e6158bcb"
)
EXPECTED_REQUIREMENTS_LOCK_SHA256 = (
    "aca8c2e21202f25d9132cf2ef14132bbc12db05c9f775635869f7c501320d335"
)
EXPECTED_DEPLOYMENT_IMAGE_MANIFEST_DIGEST = (
    "sha256:a074fac67aa01841fee592d00bae14d25dcaf98ef6e12a683ecceb7e0147e2d1"
)


def _strict_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"runtime artifact is not an object: {path}")
    return value


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_distributions() -> dict[str, tuple[str, str]]:
    locked: dict[str, tuple[str, str]] = {}
    pattern = re.compile(
        r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ]+) "
        r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
    )
    for raw_line in REQUIREMENTS_LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise RuntimeError("requirements lock contains a non-exact entry")
        name = _normalized_distribution_name(match.group("name"))
        descriptor = (match.group("version"), match.group("sha256"))
        if name in locked:
            raise RuntimeError("requirements lock contains a duplicate distribution")
        locked[name] = descriptor
    return locked


def _dependency_graph_sha256(
    locked: dict[str, tuple[str, str]],
) -> str:
    versions = {name: descriptor[0] for name, descriptor in sorted(locked.items())}
    raw = json.dumps(versions, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def verify_static_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify immutable bytes and CI use without assuming the local host platform."""

    raw_matrix = MATRIX_PATH.read_bytes()
    if hashlib.sha256(raw_matrix).hexdigest() != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("frozen runtime matrix SHA-256 mismatch")
    raw_lock = LOCK_PATH.read_bytes()
    if hashlib.sha256(raw_lock).hexdigest() != EXPECTED_LOCK_SHA256:
        raise RuntimeError("frozen runtime lock SHA-256 mismatch")
    raw_requirements = REQUIREMENTS_LOCK_PATH.read_bytes()
    if (
        hashlib.sha256(raw_requirements).hexdigest()
        != EXPECTED_REQUIREMENTS_LOCK_SHA256
    ):
        raise RuntimeError("frozen requirements lock SHA-256 mismatch")

    matrix = _strict_object(MATRIX_PATH)
    runtime_lock = _strict_object(LOCK_PATH)
    manifest = _strict_object(CANDIDATE_MANIFEST_PATH)
    locked = _locked_distributions()
    model_versions = {
        descriptor.get("xgboost_version")
        for descriptor in dict(manifest.get("artifacts") or {}).values()
        if isinstance(descriptor, dict) and "xgboost_version" in descriptor
    }
    expected_matrix_keys = {
        "candidate_id",
        "deployment_image_manifest_digest",
        "implementation",
        "lineage_id",
        "numpy_version",
        "python_version",
        "requirements_lock_sha256",
        "runtime_lock_sha256",
        "schema_version",
        "scipy_version",
        "xgboost_version",
    }
    expected_lock_keys = {
        "base_image_distributions",
        "deployment_image",
        "platform",
        "python",
        "repository_distribution",
        "requirements",
        "schema_version",
    }
    image = dict(runtime_lock.get("deployment_image") or {})
    target = dict(runtime_lock.get("platform") or {})
    requirements = dict(runtime_lock.get("requirements") or {})
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    pinned_image = (
        "python:3.12.4-slim-bookworm@"
        f"{EXPECTED_DEPLOYMENT_IMAGE_MANIFEST_DIGEST}"
    )
    if not (
        set(matrix) == expected_matrix_keys
        and matrix["schema_version"]
        == "bigan-btc-15m-residual-promotion-runtime-matrix-v2"
        and matrix["implementation"] == executor.PRODUCTION_PYTHON_IMPLEMENTATION
        and matrix["python_version"] == executor.PRODUCTION_PYTHON_VERSION
        and matrix["python_version"] == PYTHON_VERSION_PATH.read_text().strip()
        and matrix["numpy_version"] == executor.PRODUCTION_NUMPY_VERSION
        and matrix["scipy_version"] == executor.PRODUCTION_SCIPY_VERSION
        and matrix["xgboost_version"] == executor.PRODUCTION_XGBOOST_VERSION
        and matrix["runtime_lock_sha256"] == EXPECTED_LOCK_SHA256
        and matrix["requirements_lock_sha256"]
        == EXPECTED_REQUIREMENTS_LOCK_SHA256
        and matrix["deployment_image_manifest_digest"]
        == EXPECTED_DEPLOYMENT_IMAGE_MANIFEST_DIGEST
        and set(runtime_lock) == expected_lock_keys
        and runtime_lock["schema_version"]
        == "bigan-btc-15m-residual-promotion-runtime-lock-v2"
        and runtime_lock["python"]
        == {
            "implementation": matrix["implementation"],
            "version": matrix["python_version"],
        }
        and runtime_lock["base_image_distributions"] == {"pip": "24.0"}
        and runtime_lock["repository_distribution"]
        == {"name": "bigan", "version": "0.1.0"}
        and image
        == {
            "config_digest": (
                "sha256:c127439b2798d6872c53642bd6577046a30e818576250c329e55b124948dadf2"
            ),
            "index_digest": (
                "sha256:a3e58f9399353be051735f09be0316bfdeab571a5c6a24fd78b92df85bcb2d85"
            ),
            "platform_manifest_digest": EXPECTED_DEPLOYMENT_IMAGE_MANIFEST_DIGEST,
            "repository": "docker.io/library/python",
            "tag": "3.12.4-slim-bookworm",
        }
        and target
        == {
            "architecture": "x86_64",
            "libc": "glibc",
            "libc_version": "2.36",
            "operating_system": "linux",
            "python_abi": "cp312",
            "userspace_distribution": "debian",
            "userspace_version": "12",
        }
        and requirements
        == {
            "all_direct_and_transitive_artifacts_hash_locked": True,
            "dependency_graph_sha256": _dependency_graph_sha256(locked),
            "package_count": len(locked),
            "repository_path": str(REQUIREMENTS_LOCK_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_REQUIREMENTS_LOCK_SHA256,
        }
        and len(locked) == 53
        and locked["numpy"][0] == matrix["numpy_version"]
        and locked["scipy"][0] == matrix["scipy_version"]
        and locked["xgboost"][0] == matrix["xgboost_version"]
        and model_versions == {matrix["xgboost_version"]}
        and pinned_image in workflow
        and "BIGAN_DEPLOYMENT_IMAGE_MANIFEST_DIGEST" not in workflow
        and "tests/v8/test_residual_promotion_micro_live_executor.py" in workflow
        and (
            "tests/v8/test_residual_promotion_v1.py::"
            "test_repository_local_bundle_loads_and_matches_frozen_parity"
        )
        in workflow
        and "pip install --require-hashes --only-binary=:all:" in workflow
        and "pip install --no-deps --no-build-isolation -e ." in workflow
        and 'pip install -e ".[dev]"' not in workflow
    ):
        raise RuntimeError("frozen deployment runtime contract is mismatched")
    return matrix, runtime_lock


def verify_deployment_runtime() -> None:
    """Verify the exact running deployment container and every locked package."""

    matrix, runtime_lock = verify_static_contract()
    target = dict(runtime_lock["platform"])
    libc_name, libc_version = platform.libc_ver()
    os_release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    locked = _locked_distributions()
    installed = {
        _normalized_distribution_name(distribution.metadata["Name"]): (
            distribution.version
        )
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    mismatches = {
        name: {"expected": version, "actual": installed.get(name)}
        for name, (version, _) in locked.items()
        if installed.get(name) != version
    }
    expected_installed = {
        **{name: descriptor[0] for name, descriptor in locked.items()},
        **dict(runtime_lock["base_image_distributions"]),
        "bigan": str(runtime_lock["repository_distribution"]["version"]),
    }
    unexpected = sorted(set(installed) - set(expected_installed))
    _verify_model_runtime_versions(matrix)
    if not (
        platform.system().lower() == target["operating_system"]
        and platform.machine().lower() == target["architecture"]
        and libc_name.lower() == target["libc"]
        and libc_version == target["libc_version"]
        and os_release.get("ID") == target["userspace_distribution"]
        and os_release.get("VERSION_ID") == target["userspace_version"]
        and platform.python_implementation() == matrix["implementation"]
        and platform.python_version() == matrix["python_version"]
        and sys.implementation.cache_tag == "cpython-312"
        and str(sysconfig.get_config_var("SOABI")).startswith("cpython-312-")
        and not mismatches
        and not unexpected
    ):
        raise RuntimeError(
            "frozen deployment runtime matrix is mismatched; "
            f"package_mismatches={mismatches}; unexpected_packages={unexpected}"
        )


def _verify_model_runtime_versions(matrix: dict[str, Any]) -> None:
    if not (
        platform.python_implementation() == matrix["implementation"]
        and platform.python_version() == matrix["python_version"]
        and np.__version__ == matrix["numpy_version"]
        and scipy.__version__ == matrix["scipy_version"]
        and xgb.__version__ == matrix["xgboost_version"]
    ):
        raise RuntimeError("frozen deployment runtime matrix is mismatched")


def main() -> None:
    verify_deployment_runtime()


if __name__ == "__main__":
    if sys.argv[1:] == ["--static-only"]:
        verify_static_contract()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: verify_residual_promotion_runtime_matrix.py [--static-only]")
