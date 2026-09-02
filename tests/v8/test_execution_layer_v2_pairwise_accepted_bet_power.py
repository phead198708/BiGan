from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_pairwise_accepted_bet_power import (
    detectable_standardized_effect_size,
    load_and_validate_pairwise_accepted_bet_power_analysis_manifest,
    required_independent_market_count,
    run_pairwise_accepted_bet_power_analysis,
    validate_pairwise_accepted_bet_power_design,
)

DESIGN_PATH = Path(
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_pairwise_accepted_bet_power_v1.json"
)


def test_power_design_is_prospective_and_result_independent() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    validate_pairwise_accepted_bet_power_design(design)

    assert design["statistical_unit"] == "unique_accepted_bet_market"
    assert design["uses_current_oof_validation_or_confirmatory_pnl"] is False
    assert design["uses_realized_candidate_pnl_for_design"] is False
    assert design["result_dependent_extension_allowed"] is False
    assert design["promotion_evidence_eligible"] is False


def test_required_market_count_is_monotonic_in_power_and_effect_size() -> None:
    low_power = required_independent_market_count(
        alpha=0.05,
        power=0.8,
        standardized_effect_size=0.35,
    )
    high_power = required_independent_market_count(
        alpha=0.05,
        power=0.9,
        standardized_effect_size=0.35,
    )
    smaller_effect = required_independent_market_count(
        alpha=0.05,
        power=0.9,
        standardized_effect_size=0.25,
    )

    assert high_power > low_power
    assert smaller_effect > high_power


def test_power_report_recommends_materially_more_support_than_current_design(
    tmp_path: Path,
) -> None:
    first = run_pairwise_accepted_bet_power_analysis(
        run_id="power",
        output_dir=tmp_path,
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    second = run_pairwise_accepted_bet_power_analysis(
        run_id="power",
        output_dir=tmp_path / "second",
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    report = first["report"]

    assert report["recommended_required_accepted_unique_market_count"] == 88
    assert report["recommended_quality_valid_market_count"] == 220
    assert report["recommended_maximum_capture_attempt_count"] == 340
    assert report["recommended_quality_valid_market_count"] > 60
    assert report["current_60_market_design_has_limited_power"] is True
    assert report["uses_current_oof_validation_or_confirmatory_pnl"] is False
    assert report["source_model_candidate_eligible"] is False
    assert report["promotion_evidence_eligible"] is False
    assert first["report_sha256"] == second["report_sha256"]

    _, audit = load_and_validate_pairwise_accepted_bet_power_analysis_manifest(
        first["manifest_path"],
        first["manifest_sha256"],
    )
    assert audit["required_accepted_unique_market_count"] == 88
    assert audit["recommended_quality_valid_market_count"] == 220
    assert audit["recommended_maximum_capture_attempt_count"] == 340
    assert audit["uses_current_oof_validation_or_confirmatory_pnl"] is False


def test_power_manifest_validation_fails_closed_on_result_dependent_drift(
    tmp_path: Path,
) -> None:
    result = run_pairwise_accepted_bet_power_analysis(
        run_id="power-drift",
        output_dir=tmp_path,
        design_path=DESIGN_PATH,
        expected_design_sha256=_sha256(DESIGN_PATH),
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    manifest["result_dependent_extension_allowed"] = True
    drifted = tmp_path / "drifted-power-manifest.json"
    drifted.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="result_dependent_extension_allowed"):
        load_and_validate_pairwise_accepted_bet_power_analysis_manifest(
            drifted,
            _sha256(drifted),
        )


def test_current_30_accepted_market_design_only_detects_large_effect() -> None:
    detectable = detectable_standardized_effect_size(
        alpha=0.05,
        power=0.9,
        independent_market_count=30,
        robustness_inflation_factor=1.25,
    )

    assert detectable == pytest.approx(0.59735, abs=1e-5)
    assert detectable > 0.35


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
