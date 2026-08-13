#!/usr/bin/env python3
"""Fail closed unless the frozen promotion runtime matrix is exact."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import xgboost as xgb

from bigan.v8.polymarket import residual_promotion_micro_live_executor as executor

ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "examples/v8/residual_promotion_runtime_matrix.json"
LOCK_PATH = ROOT / "examples/v8/residual_promotion_runtime.lock.json"
PYTHON_VERSION_PATH = ROOT / ".python-version"
CANDIDATE_MANIFEST_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1/"
    "candidate_bundle/bundle_manifest.json"
)
EXPECTED_MATRIX_SHA256 = (
    "1ddc03f4b32f83661de77cd848b83cc9d2e1a3beea71533df2a6d21eade055ae"
)
EXPECTED_LOCK_SHA256 = (
    "c8c23d86e620358e96a6e461cd4f70d2f66a25a13397eb4cb49dd60ae3dd1672"
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


def main() -> None:
    raw_matrix = MATRIX_PATH.read_bytes()
    if hashlib.sha256(raw_matrix).hexdigest() != EXPECTED_MATRIX_SHA256:
        raise RuntimeError("frozen runtime matrix SHA-256 mismatch")
    raw_lock = LOCK_PATH.read_bytes()
    if hashlib.sha256(raw_lock).hexdigest() != EXPECTED_LOCK_SHA256:
        raise RuntimeError("frozen runtime lock SHA-256 mismatch")
    matrix = _strict_object(MATRIX_PATH)
    runtime_lock = _strict_object(LOCK_PATH)
    manifest = _strict_object(CANDIDATE_MANIFEST_PATH)
    model_versions = {
        descriptor.get("xgboost_version")
        for descriptor in dict(manifest.get("artifacts") or {}).values()
        if isinstance(descriptor, dict) and "xgboost_version" in descriptor
    }
    expected_keys = {
        "candidate_id",
        "implementation",
        "lineage_id",
        "numpy_version",
        "python_version",
        "runtime_lock_sha256",
        "schema_version",
        "scipy_version",
        "xgboost_version",
    }
    if not (
        set(matrix) == expected_keys
        and matrix["schema_version"]
        == "bigan-btc-15m-residual-promotion-runtime-matrix-v1"
        and matrix["implementation"]
        == executor.PRODUCTION_PYTHON_IMPLEMENTATION
        == platform.python_implementation()
        and matrix["python_version"]
        == executor.PRODUCTION_PYTHON_VERSION
        == platform.python_version()
        and matrix["python_version"] == PYTHON_VERSION_PATH.read_text().strip()
        and matrix["numpy_version"]
        == executor.PRODUCTION_NUMPY_VERSION
        == np.__version__
        and matrix["scipy_version"]
        == executor.PRODUCTION_SCIPY_VERSION
        == scipy.__version__
        and matrix["xgboost_version"]
        == executor.PRODUCTION_XGBOOST_VERSION
        == xgb.__version__
        and matrix["runtime_lock_sha256"] == EXPECTED_LOCK_SHA256
        and set(runtime_lock) == {"packages", "python", "schema_version"}
        and runtime_lock["schema_version"]
        == "bigan-btc-15m-residual-promotion-runtime-lock-v1"
        and runtime_lock["python"]
        == {
            "implementation": matrix["implementation"],
            "version": matrix["python_version"],
        }
        and runtime_lock["packages"]
        == {
            "numpy": matrix["numpy_version"],
            "scipy": matrix["scipy_version"],
            "xgboost": matrix["xgboost_version"],
        }
        and model_versions == {matrix["xgboost_version"]}
        and sys.version_info[:2] == (3, 12)
    ):
        raise RuntimeError("frozen deployment runtime matrix is mismatched")


if __name__ == "__main__":
    main()
