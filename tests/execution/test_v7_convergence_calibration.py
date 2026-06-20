"""Tests for v7 convergence calibration entry gating."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bigan.execution.v7_convergence_calibration import (
    V7ConvergenceCalibrationConfig,
    V7ConvergenceCalibrationGate,
)


def _write_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "calibration_summary": {
                    "sample_count": 100,
                    "hit_5c_rate": 0.55,
                    "hit_10c_rate": 0.30,
                    "close_rate": 0.45,
                    "median_best_move": 0.05,
                    "median_close_move": -0.01,
                    "median_value_error": -0.30,
                    "model_over_error_p80": 0.40,
                },
                "calibration_tables": {
                    "price_raw_edge_model": [
                        {
                            "key": ["0.40-0.50", ">=0.65", "0.40-0.50", ">=0.80"],
                            "sample_count": 30,
                            "hit_5c_rate": 0.20,
                            "hit_10c_rate": 0.05,
                            "close_rate": 0.20,
                            "median_best_move": 0.01,
                            "median_close_move": -0.08,
                            "median_value_error": -0.35,
                            "model_over_error_p80": 0.50,
                        }
                    ],
                    "price": [
                        {
                            "key": ["0.50-0.70"],
                            "sample_count": 40,
                            "hit_5c_rate": 0.80,
                            "hit_10c_rate": 0.60,
                            "close_rate": 0.50,
                            "median_best_move": 0.11,
                            "median_close_move": 0.02,
                            "median_value_error": -0.10,
                            "model_over_error_p80": 0.20,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_v7_convergence_calibration_blocks_low_hit_bucket(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.30,
            min_hit_10c_rate=0.10,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.44,
        model_value=0.83,
        edge=0.42,
        raw_p_side=0.66,
    )

    assert evaluation.matched_table == "price_raw_edge_model"
    assert evaluation.key == ("0.40-0.50", ">=0.65", "0.40-0.50", ">=0.80")
    assert evaluation.skip_reason == "calibrated_hit_5c_below_min"
    assert evaluation.to_log_payload()["stats"]["sample_count"] == 30


def test_v7_convergence_calibration_uses_hierarchy_fallback(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.70,
            min_hit_10c_rate=0.50,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.55,
        model_value=0.78,
        edge=0.23,
        raw_p_side=0.58,
    )

    assert evaluation.matched_table == "price"
    assert evaluation.key == ("0.50-0.70",)
    assert evaluation.skip_reason is None
    assert evaluation.adjusted_model_value_median == pytest.approx(0.68)
    assert evaluation.adjusted_edge_median == pytest.approx(0.13)
    assert evaluation.adjusted_model_value_p80 == pytest.approx(0.58)
    assert evaluation.adjusted_edge_p80 == pytest.approx(0.03)


def test_v7_convergence_calibration_falls_back_to_global(tmp_path: Path) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.50,
            min_hit_10c_rate=0.20,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.32,
        model_value=0.72,
        edge=0.40,
        raw_p_side=0.61,
    )

    assert evaluation.matched_table == "global"
    assert evaluation.key == ("GLOBAL",)
    assert evaluation.skip_reason is None


def test_v7_convergence_calibration_blocks_low_adjusted_median_edge(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.70,
            min_hit_10c_rate=0.50,
            min_adjusted_median_edge=0.15,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.55,
        execution_price=0.55,
        model_value=0.78,
        edge=0.23,
        raw_p_side=0.58,
    )

    assert evaluation.matched_table == "price"
    assert evaluation.adjusted_model_value_median == pytest.approx(0.68)
    assert evaluation.adjusted_edge_median == pytest.approx(0.13)
    assert evaluation.skip_reason == "calibrated_median_edge_below_min"


def test_v7_convergence_calibration_blocks_low_adjusted_p80_edge(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.70,
            min_hit_10c_rate=0.50,
            min_adjusted_p80_edge=0.05,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.55,
        execution_price=0.55,
        model_value=0.78,
        edge=0.23,
        raw_p_side=0.58,
    )

    assert evaluation.matched_table == "price"
    assert evaluation.adjusted_model_value_p80 == pytest.approx(0.58)
    assert evaluation.adjusted_edge_p80 == pytest.approx(0.03)
    assert evaluation.skip_reason == "calibrated_p80_edge_below_min"


def test_v7_convergence_calibration_adjusted_edge_uses_execution_price(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "calibration.json"
    _write_artifact(artifact)
    gate = V7ConvergenceCalibrationGate.from_json_path(
        artifact,
        config=V7ConvergenceCalibrationConfig(
            path=str(artifact),
            min_hit_5c_rate=0.70,
            min_hit_10c_rate=0.50,
            min_adjusted_median_edge=0.10,
            min_bucket_sample_count=20,
        ),
    )

    evaluation = gate.evaluate(
        price=0.55,
        execution_price=0.60,
        model_value=0.78,
        edge=0.23,
        raw_p_side=0.58,
    )

    payload = evaluation.to_log_payload()
    assert evaluation.matched_table == "price"
    assert evaluation.price == 0.55
    assert evaluation.execution_price == 0.60
    assert evaluation.adjusted_edge_median == pytest.approx(0.08)
    assert evaluation.skip_reason == "calibrated_median_edge_below_min"
    assert payload["execution_price"] == 0.60
    assert payload["adjusted_edge_median"] == pytest.approx(0.08)
