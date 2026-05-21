"""Champion/challenger shadow comparison tests for issue #45."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bigan.mlops import (
    distribution_kl_divergence,
    distribution_wasserstein_distance,
    run_shadow_comparison,
    run_shadow_warehouse_comparison,
    save_shadow_report,
)


class FakeModel:
    def __init__(self, offset: float = 0.0, fail_on_symbol: str | None = None) -> None:
        self.offset = offset
        self.fail_on_symbol = fail_on_symbol

    def predict_proba(self, row: dict) -> float:
        if row.get("source_symbol") == self.fail_on_symbol:
            raise RuntimeError("shadow model failed")
        return max(0.0, min(1.0, float(row["base_prob"]) + self.offset))


def _rows() -> list[dict]:
    return [
        {"feature_ts": 1_000, "source_symbol": "tok-1", "base_prob": 0.40},
        {"feature_ts": 2_000, "source_symbol": "tok-2", "base_prob": 0.55},
        {"feature_ts": 3_000, "source_symbol": "tok-3", "base_prob": 0.70},
    ]


def test_shadow_comparison_scores_both_models_and_reports_distribution_metrics() -> None:
    report = run_shadow_comparison(
        champion_model=FakeModel(),
        challenger_model=FakeModel(offset=0.05),
        feature_rows=_rows(),
        champion_model_version="champion-v1",
        challenger_model_version="challenger-v2",
        bins=5,
    )

    assert report.sample_count == 3
    assert report.scored_count == 3
    assert report.challenger_error_count == 0
    assert report.mean_abs_probability_delta == pytest.approx(0.05)
    assert report.max_abs_probability_delta == pytest.approx(0.05)
    assert report.kl_divergence is not None
    assert report.wasserstein_distance == pytest.approx(0.05)
    assert report.avg_challenger_latency_ms is not None
    assert report.rows[0].probability_delta == pytest.approx(0.05)


def test_shadow_errors_do_not_block_champion_predictions() -> None:
    report = run_shadow_comparison(
        champion_model=FakeModel(),
        challenger_model=FakeModel(offset=0.05, fail_on_symbol="tok-2"),
        feature_rows=_rows(),
        champion_model_version="champion-v1",
        challenger_model_version="challenger-v2",
    )

    assert report.sample_count == 3
    assert report.scored_count == 2
    assert report.challenger_error_count == 1
    failed = [row for row in report.rows if row.challenger_error]
    assert len(failed) == 1
    assert failed[0].source_symbol == "tok-2"
    assert failed[0].champion_prob_up_15m == pytest.approx(0.55)


def test_shadow_report_can_be_saved_for_promotion_review(tmp_path: Path) -> None:
    report = run_shadow_comparison(
        champion_model=FakeModel(),
        challenger_model=FakeModel(offset=-0.02),
        feature_rows=_rows(),
        champion_model_version="champion-v1",
        challenger_model_version="challenger-v2",
    )
    output_path = tmp_path / "shadow_report.json"

    save_shadow_report(report, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["champion_model_version"] == "champion-v1"
    assert saved["challenger_model_version"] == "challenger-v2"
    assert len(saved["rows"]) == 3


def test_shadow_warehouse_comparison_loads_models_and_features(tmp_path: Path) -> None:
    warehouse_dir = tmp_path / "warehouse"
    feature_dir = warehouse_dir / "features_15m_v1"
    feature_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "feature_ts": 1_000,
                    "source": "polymarket",
                    "source_symbol": "tok-1",
                    "quality_filter_pass": True,
                    "data_gap_flag": False,
                    "base_prob": 0.40,
                },
                {
                    "feature_ts": 2_000,
                    "source": "polymarket",
                    "source_symbol": "tok-2",
                    "quality_filter_pass": True,
                    "data_gap_flag": False,
                    "base_prob": 0.60,
                },
                {
                    "feature_ts": 3_000,
                    "source": "polymarket",
                    "source_symbol": "tok-gap",
                    "quality_filter_pass": True,
                    "data_gap_flag": True,
                    "base_prob": 0.90,
                },
            ]
        ),
        feature_dir / "part.parquet",
    )
    champion_path = tmp_path / "champion.json"
    challenger_path = tmp_path / "challenger.json"
    _write_logistic_model(champion_path, model_version="logreg-champion", intercept=0.0)
    _write_logistic_model(challenger_path, model_version="logreg-v3-shadow", intercept=0.2)
    output_path = tmp_path / "shadow.json"

    report = run_shadow_warehouse_comparison(
        warehouse_dir=warehouse_dir,
        champion_model_path=champion_path,
        challenger_model_path=challenger_path,
        output_path=output_path,
        since_ms=1_000,
        until_ms=3_000,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert report.champion_model_version == "logreg-champion"
    assert report.challenger_model_version == "logreg-v3-shadow"
    assert report.sample_count == 2
    assert report.scored_count == 2
    assert report.challenger_error_count == 0
    assert report.window_start_ts == 1_000
    assert report.window_end_ts == 3_000
    assert saved["sample_count"] == 2
    assert saved["rows"][0]["source_symbol"] == "tok-1"


def test_distribution_helpers_validate_inputs() -> None:
    assert distribution_kl_divergence([0.1, 0.9], [0.2, 0.8], bins=2) >= 0
    assert distribution_wasserstein_distance([0.1, 0.9], [0.2, 0.8]) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="equal length"):
        distribution_wasserstein_distance([0.1], [0.1, 0.2])


def _write_logistic_model(path: Path, *, model_version: str, intercept: float) -> None:
    path.write_text(
        json.dumps(
            {
                "model_version": model_version,
                "feature_columns": ["base_prob"],
                "coefficients": [1.0],
                "intercept": intercept,
                "means": {"base_prob": 0.5},
                "scales": {"base_prob": 1.0},
            }
        ),
        encoding="utf-8",
    )
