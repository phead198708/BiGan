from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.recorder.public_provider import (
    PolymarketPublicHTTPRealCorpusProvider,
)
from bigan.v8.polymarket.residual_promotion_collection import (
    assert_outcome_blind,
    build_progress,
    canonical_attempt_hash,
    validate_collection_authorization,
    verify_attempt_chain,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    MAXIMUM_ATTEMPTS,
    TARGET_MARKETS,
    ResidualPromotionError,
    load_residual_promotion_runtime,
    validate_final_fit_protocol,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples/v8/polymarket_configs" / LINEAGE_ID
BUNDLE = CONFIG / "candidate_bundle/bundle_manifest.json"
SOURCE_PROTOCOL = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v4"
    / "residual_v4_challenger_slot_002_protocol.json"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _attempt(index: int, *, valid: bool = True, market_id: str | None = None) -> dict:
    return {
        "attempt_index": index,
        "attempt_id": f"attempt-{index}",
        "market_id": market_id or (f"market-{index}" if valid else None),
        "quality": {
            "quality_valid": valid,
            "invalid_reason_codes": [] if valid else ["provider_incomplete"],
        },
        "provider_health": {
            "provider_failed": not valid,
            "retry_used": False,
        },
        "decision_rows": [
            {
                "candidate_selected_action": "BUY_UP_HOLD",
                "baseline_selected_action": "NO_TRADE",
                "decision_influenced_collection": False,
            }
        ],
        "outcomes_accessed": False,
        "settlement_accessed": False,
        "pnl_accessed": False,
        "previous_attempt_hash": "0" * 64,
    }


def _chain(rows: list[dict]) -> list[dict]:
    previous = "0" * 64
    output = []
    for row in rows:
        item = deepcopy(row)
        item["previous_attempt_hash"] = previous
        item["attempt_hash"] = canonical_attempt_hash(item)
        previous = item["attempt_hash"]
        output.append(item)
    return output


def test_fixed_v4_candidate_protocol_is_exactly_carried_forward() -> None:
    source = _json(SOURCE_PROTOCOL)
    protocol = _json(CONFIG / "final_fit_protocol.json")
    validate_final_fit_protocol(protocol, repository_root=REPO_ROOT)
    for field in ("model", "feature_contract", "action_policy", "pair_coherence", "target"):
        assert protocol[field] == source[field]
    assert source["action_policy"]["fixed_acceptance_threshold"] == 0.0
    assert source["prospective_power"]["maximum_market_count"] == 2_000
    assert protocol["final_fit"]["parameter_or_threshold_search"] is False


def test_authorization_is_one_slot_and_sample_plan_is_exact() -> None:
    authorization = _json(CONFIG / "lineage_authorization.json")
    assert authorization["candidate_slot"]["maximum_slots"] == 1
    assert authorization["candidate_slot"]["candidate_id"] == CANDIDATE_ID
    assert authorization["candidate_slot"]["architecture_feature_threshold_or_hyperparameter_search_allowed"] is False
    assert authorization["carry_forward"]["historical_v4_n_max_2000_gate_changed"] is False
    assert authorization["prospective_program"]["target_quality_valid_market_count"] == TARGET_MARKETS
    assert authorization["prospective_program"]["maximum_attempts"] == MAXIMUM_ATTEMPTS
    assert authorization["prospective_program"]["interim_pnl_evaluation_allowed"] is False
    assert authorization["prospective_program"]["optional_stopping_allowed"] is False
    assert authorization["safety"] == SAFETY


def test_repository_local_bundle_loads_and_matches_frozen_parity() -> None:
    runtime = load_residual_promotion_runtime(
        manifest_path=BUNDLE,
        expected_manifest_sha256=sha256_file(BUNDLE),
        repository_root=REPO_ROOT,
    )
    parity = _json(CONFIG / "candidate_bundle/offline_live_parity_report.json")
    assert runtime.candidate_id == CANDIDATE_ID
    assert parity["prediction_and_decision_parity"] is True
    assert parity["offline_projection"] == parity["live_projection"]
    assert parity["live_projection"]["model_scored"] is True
    assert parity["live_projection"]["fail_closed"] is False
    assert parity["model_refit_performed"] is False


def test_bundle_is_fresh_clone_portable_and_independent_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative_bundle_dir = BUNDLE.parent.relative_to(REPO_ROOT)
    clone_root = tmp_path / "fresh-clone"
    destination = clone_root / relative_bundle_dir
    destination.parent.mkdir(parents=True)
    shutil.copytree(BUNDLE.parent, destination)
    monkeypatch.chdir(tmp_path)
    runtime = load_residual_promotion_runtime(
        manifest_path=clone_root / BUNDLE.relative_to(REPO_ROOT),
        expected_manifest_sha256=sha256_file(BUNDLE),
        repository_root=clone_root,
    )
    assert runtime.manifest_sha256 == sha256_file(BUNDLE)


def test_bundle_child_byte_drift_fails_closed(tmp_path: Path) -> None:
    relative_bundle_dir = BUNDLE.parent.relative_to(REPO_ROOT)
    clone_root = tmp_path / "fresh-clone"
    destination = clone_root / relative_bundle_dir
    destination.parent.mkdir(parents=True)
    shutil.copytree(BUNDLE.parent, destination)
    adapter = destination / "soft_stacker_adapter.json"
    adapter.write_bytes(adapter.read_bytes() + b"\n")
    with pytest.raises(ResidualPromotionError, match="SHA-256 mismatch"):
        load_residual_promotion_runtime(
            manifest_path=clone_root / BUNDLE.relative_to(REPO_ROOT),
            expected_manifest_sha256=sha256_file(BUNDLE),
            repository_root=clone_root,
        )


def test_outcome_blind_fields_fail_closed_without_rejecting_population_target() -> None:
    assert_outcome_blind(
        {
            "target_quality_valid_market_count": TARGET_MARKETS,
            "outcomes_accessed": False,
            "settlement_accessed": False,
            "pnl_accessed": False,
        }
    )
    for forbidden in ("outcome", "settlement", "realized_pnl", "unit_pnl", "target", "label"):
        with pytest.raises(ValueError, match="forbidden outcome-bearing field"):
            assert_outcome_blind({forbidden: 1})
    for flag in ("outcomes_accessed", "settlement_accessed", "pnl_accessed"):
        with pytest.raises(ValueError, match="must be false"):
            assert_outcome_blind({flag: True})


def test_progress_is_hash_chained_and_model_decisions_do_not_select_population() -> None:
    rows = _chain([_attempt(1), _attempt(2, valid=False), _attempt(3)])
    verify_attempt_chain(rows)
    first = build_progress(
        rows,
        authorization_sha256="a" * 64,
        collector_protocol_sha256="b" * 64,
        candidate_bundle_sha256="c" * 64,
    )
    changed = deepcopy(rows)
    changed[0]["decision_rows"][0]["candidate_selected_action"] = "NO_TRADE"
    changed[0]["decision_rows"][0]["baseline_selected_action"] = "BUY_DOWN_HOLD"
    changed = _chain(changed)
    second = build_progress(
        changed,
        authorization_sha256="a" * 64,
        collector_protocol_sha256="b" * 64,
        candidate_bundle_sha256="c" * 64,
    )
    assert first["quality_valid_market_count"] == second["quality_valid_market_count"] == 2
    assert first["remaining_quality_valid_markets"] == second["remaining_quality_valid_markets"] == 2_498
    assert first["collection_influenced_by_model_decisions"] is False
    assert first["fresh_outcomes_opened"] is False


def test_frozen_collection_authorization_and_runtime_validate() -> None:
    result = validate_collection_authorization(
        authorization_path=CONFIG / "manual_collection_authorization_v3.json",
        collector_protocol_path=CONFIG / "prospective_collector_protocol_v3.json",
        repository_root=REPO_ROOT,
    )
    assert result["validation_passed"] is True
    assert result["fresh_outcomes_opened"] is False
    assert result["runtime"].candidate_id == CANDIDATE_ID
    assert result["safety"] == SAFETY


def test_zero_attempt_collector_correction_supersedes_original_fail_closed() -> None:
    correction = _json(CONFIG / "collector_pre_attempt_engineering_correction.json")
    assert correction["detected_before_first_attempt"] is True
    assert correction["prior_attempts_consumed"] == 0
    assert correction["prior_quality_valid_market_count"] == 0
    assert correction["original_authorization_invalidated_for_execution"] is True
    assert correction["model_prediction_bytes_changed"] is False
    assert correction["threshold_gate_cost_baseline_or_population_changed"] is False
    assert correction["fresh_outcomes_accessed"] is False
    assert correction["safety"] == SAFETY
    with pytest.raises(ValueError, match="superseded"):
        validate_collection_authorization(
            authorization_path=CONFIG / "manual_collection_authorization.json",
            collector_protocol_path=CONFIG / "prospective_collector_protocol.json",
            repository_root=REPO_ROOT,
        )


def test_corrected_collector_graph_requires_chainlink_and_reconciles_hashes() -> None:
    protocol = _json(CONFIG / "prospective_collector_protocol_v3.json")
    assert protocol["chainlink_rtds_background_collector_required"] is True
    assert protocol["chainlink_rtds_injected_into_capture"] is True
    assert protocol["chainlink_missing_or_stale_behavior"] == "quality_invalid_fail_closed"
    assert protocol["prior_attempts_consumed"] == 1
    assert protocol["prior_failed_attempt_preserved"] is True
    assert protocol["full_window_coverage_measurement_required"] is True
    assert protocol["rest_fallback_collection_seconds"] == 5.0
    bindings = protocol["implementation_bindings"]
    assert set(bindings) == {
        "collector_cli",
        "collection_ledger",
        "pending_capture",
        "chainlink_rtds",
        "live_round_finalizer",
    }
    for descriptor in bindings.values():
        assert sha256_file(REPO_ROOT / descriptor["path"]) == descriptor["sha256"]
    collector_source = (
        REPO_ROOT / bindings["collector_cli"]["path"]
    ).read_text(encoding="utf-8")
    assert "PolymarketChainlinkRTDSCollector" in collector_source
    assert "chainlink_rtds_collector=chainlink" in collector_source
    assert "rest_fallback_collection_seconds" in collector_source
    provenance = _json(
        CONFIG / "collector_pre_attempt_engineering_correction.json"
    )["source_provenance"]
    assert provenance["source_commit"] == "0595b168512c43f45966957dde8b36a23723cbce"
    assert provenance["source_content_sha256"] == bindings["live_round_finalizer"]["sha256"]


def test_coverage_correction_preserves_failed_attempt_and_supersedes_v2() -> None:
    correction = _json(CONFIG / "collector_coverage_instrumentation_correction.json")
    assert correction["detected_after_attempt"] == 1
    assert correction["prior_attempts_consumed"] == 1
    assert correction["prior_quality_valid_market_count"] == 0
    assert correction["failed_attempt_preserved"] is True
    assert correction["first_attempt_audit"]["quality_valid"] is False
    assert correction["first_attempt_audit"]["resolution_rows"] == 0
    assert correction["gate_threshold_or_model_changed"] is False
    assert correction["model_prediction_bytes_changed"] is False
    assert correction["fresh_outcomes_accessed"] is False
    with pytest.raises(ValueError, match="superseded by coverage correction"):
        validate_collection_authorization(
            authorization_path=CONFIG / "manual_collection_authorization_v2.json",
            collector_protocol_path=CONFIG / "prospective_collector_protocol_v2.json",
            repository_root=REPO_ROOT,
        )


def test_v3_provider_enables_existing_full_window_coverage_measurement() -> None:
    provider = PolymarketPublicHTTPRealCorpusProvider(
        rest_fallback_collection_seconds=5.0
    )
    assert provider._full_window_coverage_required() is True


def test_prospective_protocol_preserves_economic_gates_and_no_optional_stopping() -> None:
    protocol = _json(CONFIG / "prospective_statistical_protocol.json")
    assert protocol["population"]["target_quality_valid_market_count"] == 2_500
    assert protocol["population"]["maximum_attempts"] == 3_000
    assert protocol["bootstrap"] == {
        "method": "market_level_paired_percentile_bootstrap",
        "seed": 2_642_500,
        "resamples": 10_000,
        "confidence": 0.975,
        "lower_quantile": 0.025,
        "candidate_and_baseline_share_indices": True,
        "NO_TRADE_participates_as_zero": True,
    }
    assert protocol["cost_stress_multipliers"] == [1.2, 1.5, 2.0]
    assert all(protocol["gates"].values())
    assert protocol["gates"]["stable_score_to_realized_pnl_ordering"] is True
    inheritance = protocol["gate_inheritance"]
    assert inheritance["source_v4_gates"] == _json(SOURCE_PROTOCOL)["gates"]
    assert inheritance["source_v4_prospective_power"]["maximum_market_count"] == 2_000
    assert inheritance["all_non_sample_plan_economic_and_integrity_gates_unchanged"] is True
    assert inheritance["fixed_acceptance_threshold"] == 0.0
    assert protocol["gate_implementation"] == _json(SOURCE_PROTOCOL)["inputs"]["gate_implementation"]
    for descriptor_name in (
        "candidate_bundle",
        "runtime_implementation",
        "source_v4_protocol",
        "gate_implementation",
        "baseline_contract",
        "baseline_artifact",
        "feature_contract",
        "cost_contract",
        "reporting_contract",
        "runtime_parity_report",
    ):
        descriptor = protocol[descriptor_name]
        path = REPO_ROOT / descriptor["path"]
        assert path.is_file()
        assert sha256_file(path) == descriptor["sha256"]
    assert protocol["failure_semantics"] == {
        "any_gate_failure_terminalizes_lineage": True,
        "gate_waiver_allowed": False,
        "rerun_allowed": False,
        "optional_stopping_allowed": False,
        "interim_pnl_evaluation_allowed": False,
    }
    assert protocol["safety"] == SAFETY


def test_every_frozen_json_has_matching_sidecar_and_safety_is_locked() -> None:
    for path in sorted(CONFIG.rglob("*.json")):
        sidecar = path.with_suffix(path.suffix + ".sha256")
        assert sidecar.read_text(encoding="utf-8").strip() == sha256_file(path)
        payload = _json(path)
        if "safety" in payload:
            assert payload["safety"] == SAFETY
    assert all(value is False for value in SAFETY.values())
