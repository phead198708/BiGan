from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    TARGET_MANIFEST_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_fit import (
    RuntimeAlignedSBCNetReturnV64FitConfig,
    _candidate_freeze_gate,
    _feature_matrix,
    _finite_sample_higher_quantile,
    run_runtime_aligned_sbc_net_return_v6_4_fit,
    validate_runtime_aligned_v6_4_fit_profile,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / (
    "examples/v8/polymarket_configs/"
    "execution_layer_v2_runtime_aligned_sbc_net_return_v6_4_fit_profile.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _features(value: float, *, side: str, sentinel: bool = False) -> dict[str, float]:
    score = -1_000_000.0 if sentinel else 0.2 + value * 0.05
    return {
        "side_is_up": float(side == "UP"),
        "execution_price": 0.5 + value * 0.02,
        "current_bid": 0.48 + value * 0.02,
        "spread_bps": 200.0 + value,
        "book_staleness_ms": 100.0 + value,
        "queue_fill_probability_proxy": 0.9,
        "liquidity_depth_log1p": 5.0 + value * 0.01,
        "executable_ask_notional_log1p": 4.0 + value * 0.01,
        "executable_bid_notional_log1p": 4.2 + value * 0.01,
        "time_to_close_seconds": 240.0 - value,
        "recent_book_update_count_1m": 4.0 + value * 0.1,
        "recent_bid_depth_volatility_1m": 0.01 + abs(value) * 0.001,
        "recent_spread_stability_1m": 0.9,
        "combined_spread_bps": 400.0 + value,
        "liquidity_imbalance": value * 0.02,
        "btc_return_30s": value * 0.001,
        "btc_return_1m": value * 0.002,
        "reference_price_to_beat_distance_at_decision": value * 0.0015,
        "canonical_v6_2_score": score,
        "action_score_margin": score,
        "selected_side_probability": 0.52 + value * 0.005,
        "pre_entry_market_exposure": 0.0,
        "same_side_prior_entry": 0.0,
        "side_flip_prior_entry": 0.0,
    }


def _synthetic_lineage(tmp_path: Path) -> tuple[Path, Path]:
    target_profile = tmp_path / "target_profile.json"
    lineage_freeze = tmp_path / "lineage_freeze.json"
    _write_json(target_profile, {"frozen": True})
    _write_json(lineage_freeze, {"lineage_freeze_passed": True})
    rows = []
    for role, market_count, start in (
        ("development_train", 89, 1_000_000),
        ("development_calibration", 45, 100_000_000),
    ):
        for market_index in range(market_count):
            market_id = f"{role}-{market_index:03d}"
            market_value = ((market_index % 11) - 5) / 10.0
            for slot in range(4):
                for side in ("UP", "DOWN"):
                    value = market_value + slot * 0.04 + (0.08 if side == "UP" else -0.03)
                    decision_ts = start + market_index * 100_000 + slot * 10_000
                    target = 0.12 + 0.28 * value + (0.035 if side == "UP" else 0.0)
                    rows.append(
                        {
                            "market_id": market_id,
                            "role": role,
                            "side": side,
                            "action": f"BUY_{side}_SELL_BEFORE_CLOSE",
                            "decision_ts": decision_ts,
                            "max_input_ts": decision_ts,
                            "features": _features(value, side=side),
                            "runtime_policy_after_cost_net_pnl_per_contract": target,
                        }
                    )
    rows_path = tmp_path / "runtime_rows.jsonl"
    _write_jsonl(rows_path, rows)
    target_manifest = {
        "schema_version": TARGET_MANIFEST_SCHEMA_VERSION,
        "target_corpus_gate_passed": True,
        "runtime_aligned_rows": {"path": str(rows_path), "sha256": _sha(rows_path)},
        "profile": {"path": str(target_profile), "sha256": _sha(target_profile)},
        "lineage_freeze_manifest": {
            "path": str(lineage_freeze),
            "sha256": _sha(lineage_freeze),
        },
    }
    target_manifest_path = tmp_path / "target_manifest.json"
    _write_json(target_manifest_path, target_manifest)
    profile = json.loads(PROFILE_PATH.read_text())
    profile["source_lineage"] = {
        "target_manifest_sha256": _sha(target_manifest_path),
        "target_rows_sha256": _sha(rows_path),
        "target_profile_sha256": _sha(target_profile),
        "lineage_freeze_manifest_sha256": _sha(lineage_freeze),
        "v6_2_source_model_sha256": "a" * 64,
    }
    fit_profile_path = tmp_path / "fit_profile.json"
    _write_json(fit_profile_path, profile)
    return target_manifest_path, fit_profile_path


def test_v6_4_fit_profile_rejects_threshold_search() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    validate_runtime_aligned_v6_4_fit_profile(profile)
    profile["calibration"]["threshold_search_enabled"] = True
    with pytest.raises(ValueError, match="calibration"):
        validate_runtime_aligned_v6_4_fit_profile(profile)


def test_v6_4_sentinel_transform_is_explicit_and_deterministic() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    model = profile["model"]
    rows = [
        {"features": _features(0.1, side="UP", sentinel=True)},
        {"features": _features(0.2, side="DOWN", sentinel=False)},
    ]
    matrix, names = _feature_matrix(
        rows, model["sentinel_feature_policy"], model["raw_feature_columns"]
    )
    score_index = names.index("canonical_v6_2_score")
    available_index = names.index("canonical_v6_2_score_available")
    assert matrix[0, score_index] == 0.0
    assert matrix[0, available_index] == 0.0
    assert matrix[1, available_index] == 1.0
    assert _finite_sample_higher_quantile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_v6_4_fit_freezes_only_after_all_fixed_gates_pass(tmp_path: Path) -> None:
    target_manifest_path, fit_profile_path = _synthetic_lineage(tmp_path)
    result = run_runtime_aligned_sbc_net_return_v6_4_fit(
        RuntimeAlignedSBCNetReturnV64FitConfig(
            run_id="synthetic-v6-4-fit",
            output_dir=tmp_path / "runs",
            fit_profile_path=fit_profile_path,
            expected_fit_profile_sha256=_sha(fit_profile_path),
            target_manifest_path=target_manifest_path,
            expected_target_manifest_sha256=_sha(target_manifest_path),
            implementation_commit="b" * 40,
        )
    )
    report = result["report"]
    manifest = result["manifest"]
    assert report["calibration_gate_passed"] is True
    assert report["candidate_freeze_gate_passed"] is True
    assert report["positive_lcb_unique_market_count_by_side"]["UP"] >= 3
    assert report["positive_lcb_unique_market_count_by_side"]["DOWN"] >= 3
    assert manifest["candidate_scoring_frozen"] is True
    assert manifest["outcome_blind_future_collection_resume_allowed"] is True
    assert manifest["source_model_candidate_eligible"] is False
    assert manifest["freeze_ready"] is False
    assert manifest["promotion_evidence_eligible"] is False
    assert manifest["#134_resume_allowed"] is False
    assert manifest["#146_start_allowed"] is False
    assert manifest["historical_oof_opened"] is False


def test_v6_4_freeze_gate_fails_closed_when_calibration_fails() -> None:
    profile = json.loads(PROFILE_PATH.read_text())
    gate = _candidate_freeze_gate(
        [],
        model={"coefficients_finite": True, "coefficients_bounded": True},
        calibration={
            "calibration_gate_passed": False,
            "calibration_gate_blocking_reason_codes": ["relative_mae_improvement_gate_failed"],
        },
        stability={"coefficient_stability_gate_passed": True},
        profile=profile,
    )
    assert gate["candidate_freeze_gate_passed"] is False
    assert "relative_mae_improvement_gate_failed" in gate[
        "candidate_freeze_blocking_reason_codes"
    ]
