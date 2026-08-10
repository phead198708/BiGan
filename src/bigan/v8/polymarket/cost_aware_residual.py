"""Governed rolling-origin development for the BTC 15m cost-aware residual."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.challenge_model_15m_training import (
    BASE_FEATURE_NAMES,
    SIDES,
    side_symmetric_features,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import FEATURE_NAMES
from bigan.v8.polymarket.moe_confirmatory_evaluation import (
    _evaluation_artifacts,
    _load_exact_contexts,
    _write_new_frozen_text,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-cost-aware-market-residual-v1"
PROTOCOL_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-oof-protocol-v1"
PREDICTION_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-oof-prediction-v1"
FOLD_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-oof-fold-v1"
MARKET_RESULT_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-market-result-v1"
REPORT_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-oof-report-v1"
MANIFEST_SCHEMA_VERSION = "bigan-btc-15m-cost-aware-residual-oof-manifest-v1"

DEFAULT_CONFIG_DIR = (
    REPO_ROOT
    / "examples/v8/polymarket_configs/BTC-15M-cost-aware-market-residual-v1"
)
DEFAULT_PROTOCOL = DEFAULT_CONFIG_DIR / "residual_primary_slot_001_protocol.json"
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "residual_primary_slot_001_oof"
PARENT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
)


def validate_residual_oof_protocol(
    payload: Mapping[str, Any],
    *,
    repository_root: Path | str = REPO_ROOT,
    verify_artifacts: bool = True,
) -> None:
    """Fail closed unless the single candidate is fully preregistered."""

    blockers: list[str] = []
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        blockers.append("schema_version")
    if payload.get("lineage_id") != LINEAGE_ID:
        blockers.append("lineage_id")
    if payload.get("candidate_role") != "primary":
        blockers.append("candidate_role")
    if payload.get("development_only_forever") is not True:
        blockers.append("development_only_forever")
    if payload.get("promotion_evidence_eligible") is not False:
        blockers.append("promotion_evidence_eligible")
    budget = dict(payload.get("candidate_budget") or {})
    if (
        budget.get("maximum_total_slots") != 2
        or budget.get("this_slot_ordinal") != 1
        or budget.get("slots_consumed_before_run") != 0
        or budget.get("slot_budget_may_be_increased") is not False
    ):
        blockers.append("candidate_budget")
    target = dict(payload.get("target") or {})
    if target != {
        "name": "direct_after_cost_action_value",
        "formula": (
            "settlement_payout-executable_ask-frozen_fees-slippage-"
            "liquidity_impact"
        ),
        "execution_policy": "HOLD_TO_SETTLEMENT",
        "NO_TRADE_value": 0.0,
        "post_close_only": True,
    }:
        blockers.append("target")
    feature = dict(payload.get("feature_contract") or {})
    if (
        feature.get("ordered_feature_count") != 108
        or feature.get("base_feature_count") != 54
        or feature.get("shared_side_symmetric_model") is not True
        or feature.get("side_identity_feature_allowed") is not False
        or feature.get("native_missing_value") != "nan"
        or feature.get("missing_values_encoded_as_zero") is not False
        or feature.get("feature_search_allowed") is not False
        or feature.get("market_horizon_seconds") != 900
    ):
        blockers.append("feature_contract")
    model = dict(payload.get("model") or {})
    parameters = dict(model.get("parameters") or {})
    if (
        model.get("family") != "pooled_global_xgboost_direct_regressor"
        or model.get("route_or_expert_allowed") is not False
        or model.get("fixed_num_boost_round") is None
        or int(model.get("fixed_num_boost_round") or 0) <= 0
        or parameters.get("objective") != "reg:squarederror"
        or parameters.get("eval_metric") != "rmse"
        or parameters.get("tree_method") != "hist"
        or parameters.get("nthread") != 1
    ):
        blockers.append("model")
    rolling = dict(payload.get("rolling_origin") or {})
    if (
        rolling.get("market_order") != "frozen_capture_order"
        or rolling.get("initial_training_market_count") != 200
        or rolling.get("target_block_size") != 100
        or rolling.get("target_block_count") != 6
        or rolling.get("oof_market_count") != 600
        or rolling.get("strictly_prior_market_labels_only") is not True
        or rolling.get("market_grouped") is not True
        or rolling.get("future_market_used_for_fit") is not False
    ):
        blockers.append("rolling_origin")
    action = dict(payload.get("action_policy") or {})
    if action != {
        "decision_order": "chronological",
        "accept_if": "highest_side_prediction>0",
        "fixed_acceptance_threshold": 0.0,
        "side_tie_break_order": ["UP", "DOWN"],
        "one_trade_maximum_per_market": True,
        "NO_TRADE_if_no_positive_prediction": True,
        "NO_TRADE_unit_pnl": 0.0,
        "threshold_search_allowed": False,
    }:
        blockers.append("action_policy")
    bootstrap = dict(payload.get("bootstrap") or {})
    if (
        bootstrap.get("method") != "market_level_paired_percentile_bootstrap"
        or bootstrap.get("confidence") != 0.975
        or bootstrap.get("lower_quantile") != 0.025
        or int(bootstrap.get("resamples") or 0) <= 0
        or bootstrap.get("candidate_and_baseline_share_indices") is not True
        or bootstrap.get("NO_TRADE_participates_as_zero") is not True
    ):
        blockers.append("bootstrap")
    stress = dict(payload.get("cost_stress") or {})
    if (
        stress.get("multipliers") != [1.2, 1.5, 2.0]
        or stress.get("action_selection_reused_from_base_cost") is not True
        or stress.get("formula")
        != "gross_price_edge-multiplier*total_cost_relative_to_mid"
    ):
        blockers.append("cost_stress")
    gates = dict(payload.get("gates") or {})
    required_gates = {
        "absolute_market_bootstrap_97_5pct_lcb_gt_zero",
        "paired_delta_market_bootstrap_97_5pct_lcb_gt_zero",
        "every_chronological_block_candidate_total_gte_zero",
        "every_chronological_block_paired_delta_total_gte_zero",
        "largest_winner_removed_candidate_total_gte_zero",
        "largest_positive_delta_removed_total_gte_zero",
        "stable_score_to_realized_pnl_ordering",
        "all_cost_stress_candidate_totals_gte_zero",
        "all_cost_stress_paired_delta_totals_gte_zero",
        "prospective_power_required_market_count_lte_2000",
        "population_and_leakage_reconciliation",
    }
    if set(gates) != required_gates or any(value is not True for value in gates.values()):
        blockers.append("gates")
    power = dict(payload.get("prospective_power") or {})
    if (
        power.get("confidence") != 0.975
        or power.get("target_power") != 0.8
        or power.get("effect_haircut") != 0.5
        or power.get("maximum_market_count") != 2000
        or power.get("required_n_rule")
        != "max_absolute_and_paired_plugin_normal_approximation"
    ):
        blockers.append("prospective_power")
    discipline = dict(payload.get("development_discipline") or {})
    if not (
        discipline.get("one_candidate_this_slot") is True
        and discipline.get("hyperparameter_search_allowed") is False
        and discipline.get("threshold_search_allowed") is False
        and discipline.get("route_side_missingness_or_outlier_filtering_allowed")
        is False
        and discipline.get("post_result_mutation_allowed") is False
        and discipline.get("challenger_requires_separate_preregistration") is True
    ):
        blockers.append("development_discipline")
    state = dict(payload.get("state") or {})
    if state != {
        "training_started": False,
        "candidate_frozen": False,
        "live_shadow_started": False,
        "fresh_confirmatory_collection_started": False,
        "fresh_outcomes_opened": False,
    }:
        blockers.append("state")
    if dict(payload.get("safety") or {}) != SAFETY:
        blockers.append("safety")
    inputs = dict(payload.get("inputs") or {})
    required_inputs = {
        "lineage_authorization",
        "development_data_registry",
        "raw_capture_recovery_bundle_manifest",
        "terminal_diagnostic_scored_rows",
        "confirmatory_capture_manifest",
        "confirmatory_market_evaluation_rows",
        "baseline_decision_rows",
        "matched_global_baseline_contract",
        "parent_feature_contract",
        "parent_cost_and_action_contract",
        "implementation",
    }
    if set(inputs) != required_inputs:
        blockers.append("inputs")
    root = Path(repository_root).resolve()
    if verify_artifacts and not blockers:
        for name, descriptor in inputs.items():
            try:
                _verify_descriptor(dict(descriptor), repository_root=root)
            except (KeyError, OSError, TypeError, ValueError):
                blockers.append(f"inputs.{name}")
        authorization = _load_json(
            _verify_descriptor(inputs["lineage_authorization"], repository_root=root)
        )
        authorization_scope = dict(authorization.get("authorization_scope") or {})
        if not (
            authorization.get("lineage_id") == LINEAGE_ID
            and authorization_scope.get("outcome_aware_development_training_authorized")
            is True
            and authorization_scope.get("training_may_start_only_after_slot_protocol_is_sha_frozen")
            is True
            and authorization_scope.get("candidate_slot_budget", {}).get(
                "maximum_total_slots"
            )
            == 2
        ):
            blockers.append("lineage_authorization")
    if blockers:
        raise ValueError("residual OOF protocol invalid: " + ", ".join(blockers))


def run_residual_rolling_origin_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    expected_protocol_sha256: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    source_commit: str,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Execute the single frozen primary candidate exactly once."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    if not protocol_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("residual OOF paths must remain repository-local")
    if sha256_file(protocol_file) != expected_protocol_sha256.lower():
        raise ValueError("residual OOF protocol SHA-256 mismatch")
    sidecar = protocol_file.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != expected_protocol_sha256:
        raise ValueError("residual OOF protocol is not SHA-frozen")
    if not _looks_like_git_sha(source_commit):
        raise ValueError("source_commit must be a full lowercase Git SHA")
    protocol = _load_json(protocol_file)
    validate_residual_oof_protocol(protocol, repository_root=root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"residual OOF output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    dataset_rows, baseline_by_market, population_order = _load_frozen_development_rows(
        protocol=protocol,
        repository_root=root,
    )
    predictions, folds = _rolling_origin_predict(
        rows=dataset_rows,
        population_order=population_order,
        protocol=protocol,
    )
    market_results = market_results_from_predictions(
        predictions=predictions,
        baseline_by_market=baseline_by_market,
        population_order=population_order,
        initial_training_market_count=int(
            protocol["rolling_origin"]["initial_training_market_count"]
        ),
        target_block_size=int(protocol["rolling_origin"]["target_block_size"]),
    )
    report = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=expected_protocol_sha256,
        source_commit=source_commit,
        market_results=market_results,
        fold_audits=folds,
    )

    dataset_path = output / "residual_development_dataset_rows.jsonl"
    prediction_path = output / "residual_oof_predictions.jsonl"
    fold_path = output / "residual_oof_fold_audits.jsonl"
    market_path = output / "residual_oof_market_results.jsonl"
    report_path = output / "residual_oof_report.json"
    markdown_path = output / "residual_oof_report.md"
    _write_new_jsonl(dataset_path, [_public_dataset_row(row) for row in dataset_rows])
    _write_new_jsonl(prediction_path, predictions)
    _write_new_jsonl(fold_path, folds)
    _write_new_jsonl(market_path, market_results)
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_artifact = _write_new_frozen_text(
        markdown_path,
        render_residual_oof_markdown(report),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol": _descriptor(protocol_file, root),
        "artifacts": {
            "dataset_rows": _descriptor(dataset_path, root),
            "predictions": _descriptor(prediction_path, root),
            "fold_audits": _descriptor(fold_path, root),
            "market_results": _descriptor(market_path, root),
            "report": _descriptor(Path(report_artifact["path"]), root),
            "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        },
        "evaluation_executed_exactly_once": True,
        "candidate_freeze_allowed": report["all_gates_passed"],
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "residual_oof_manifest.json",
        manifest,
    )
    return {
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "report": _descriptor(Path(report_artifact["path"]), root),
        "all_gates_passed": report["all_gates_passed"],
        "failed_gates": report["failed_gates"],
        "oof_market_count": len(market_results),
        "safety": dict(SAFETY),
    }


def verify_frozen_residual_oof(
    *,
    protocol_path: Path | str = DEFAULT_PROTOCOL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Verify hashes and independently rebuild the frozen OOF report."""

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    output = Path(output_dir).resolve()
    protocol = _verified_json(protocol_file)
    validate_residual_oof_protocol(protocol, repository_root=root)
    manifest = _verified_json(output / "residual_oof_manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("residual OOF manifest schema mismatch")
    if manifest.get("protocol") != _descriptor(protocol_file, root):
        raise ValueError("residual OOF manifest protocol binding mismatch")
    artifact_paths = {
        name: _verify_descriptor(descriptor, repository_root=root)
        for name, descriptor in dict(manifest["artifacts"]).items()
    }
    predictions = _load_jsonl(artifact_paths["predictions"])
    folds = _load_jsonl(artifact_paths["fold_audits"])
    markets = _load_jsonl(artifact_paths["market_results"])
    _validate_frozen_population(
        predictions=predictions,
        fold_audits=folds,
        market_results=markets,
        protocol=protocol,
    )
    rebuilt = build_residual_oof_report(
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_file),
        source_commit=str(manifest["source_commit"]),
        market_results=markets,
        fold_audits=folds,
    )
    frozen_report = _load_json(artifact_paths["report"])
    if rebuilt != frozen_report:
        raise ValueError("residual OOF report does not reproduce")
    if render_residual_oof_markdown(rebuilt) != artifact_paths[
        "report_markdown"
    ].read_text(encoding="utf-8"):
        raise ValueError("residual OOF markdown does not reproduce")
    return {
        "verification_passed": True,
        "all_gates_passed": rebuilt["all_gates_passed"],
        "failed_gates": rebuilt["failed_gates"],
        "oof_market_count": len(markets),
        "manifest_sha256": sha256_file(output / "residual_oof_manifest.json"),
        "safety": dict(SAFETY),
    }


def market_results_from_predictions(
    *,
    predictions: Sequence[Mapping[str, Any]],
    baseline_by_market: Mapping[str, Mapping[str, Any]],
    population_order: Sequence[str],
    initial_training_market_count: int,
    target_block_size: int,
) -> list[dict[str, Any]]:
    """Apply the frozen zero-threshold policy and keep every OOF market."""

    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        rows_by_market[str(row["market_id"])].append(row)
    oof_order = list(population_order[initial_training_market_count:])
    if set(rows_by_market) != set(oof_order) or len(rows_by_market) != len(oof_order):
        raise ValueError("candidate OOF prediction population mismatch")
    results = []
    for oof_index, market_id in enumerate(oof_order):
        candidates = sorted(
            rows_by_market[market_id],
            key=lambda row: (int(row["decision_ts"]), SIDES.index(str(row["side"]))),
        )
        selected: Mapping[str, Any] | None = None
        for decision_ts in sorted({int(row["decision_ts"]) for row in candidates}):
            at_decision = [
                row for row in candidates if int(row["decision_ts"]) == decision_ts
            ]
            if [str(row["side"]) for row in at_decision] != list(SIDES):
                raise ValueError("each candidate decision must contain ordered UP/DOWN rows")
            best = max(
                at_decision,
                key=lambda row: (float(row["prediction"]), -SIDES.index(str(row["side"]))),
            )
            if float(best["prediction"]) > 0.0:
                selected = best
                break
        baseline = dict(baseline_by_market.get(market_id) or {})
        if not baseline:
            raise ValueError("matched baseline market row missing")
        candidate_pnl = float(selected["realized_unit_net_pnl_if_action"]) if selected else 0.0
        candidate_cost = (
            _stress_cost_from_prediction(selected) if selected else 0.0
        )
        baseline_pnl = float(baseline["baseline_unit_net_pnl"])
        baseline_cost = float(
            dict(dict(baseline["cost_decomposition"])["baseline"])["total_cost"]
        )
        if baseline.get("baseline_accepted") is not True and (
            baseline_pnl != 0.0 or baseline_cost != 0.0
        ):
            raise ValueError("baseline NO_TRADE must have zero PnL and cost")
        result = {
            "schema_version": MARKET_RESULT_SCHEMA_VERSION,
            "lineage_id": LINEAGE_ID,
            "market_id": market_id,
            "market_start_ts": int(candidates[0]["market_start_ts"]),
            "oof_position": oof_index + 1,
            "chronological_block": oof_index // target_block_size + 1,
            "chronological_half": "first" if oof_index < len(oof_order) // 2 else "second",
            "candidate_accepted": selected is not None,
            "candidate_selected_side": str(selected["side"]) if selected else None,
            "candidate_decision_ts": int(selected["decision_ts"]) if selected else None,
            "candidate_prediction": float(selected["prediction"]) if selected else None,
            "candidate_unit_net_pnl": candidate_pnl,
            "candidate_total_cost_relative_to_mid": candidate_cost,
            "baseline_accepted": bool(baseline["baseline_accepted"]),
            "baseline_selected_side": baseline["baseline_selected_side"],
            "baseline_decision_ts": int(baseline["decision_ts"]),
            "baseline_unit_net_pnl": baseline_pnl,
            "baseline_total_cost_relative_to_mid": baseline_cost,
            "paired_delta_unit_net_pnl": candidate_pnl - baseline_pnl,
            "NO_TRADE_participates_as_zero": True,
            "development_only_forever": True,
            "promotion_evidence_eligible": False,
            "safety": dict(SAFETY),
        }
        results.append(result)
    return results


def build_residual_oof_report(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    source_commit: str,
    market_results: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute all preregistered gates from market-level OOF rows."""

    _validate_market_results(market_results, protocol=protocol)
    _validate_fold_audits(fold_audits, protocol=protocol)
    candidate = np.asarray(
        [float(row["candidate_unit_net_pnl"]) for row in market_results], dtype=float
    )
    baseline = np.asarray(
        [float(row["baseline_unit_net_pnl"]) for row in market_results], dtype=float
    )
    delta = candidate - baseline
    bootstrap = dict(protocol["bootstrap"])
    indices = _bootstrap_indices(
        market_count=len(market_results),
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    confidence = float(bootstrap["confidence"])
    candidate_interval = _bootstrap_interval(candidate, indices=indices, confidence=confidence)
    baseline_interval = _bootstrap_interval(baseline, indices=indices, confidence=confidence)
    delta_interval = _bootstrap_interval(delta, indices=indices, confidence=confidence)
    candidate_lwr = float(candidate.sum() - candidate[int(np.argmax(candidate))])
    delta_lwr = float(delta.sum() - delta[int(np.argmax(delta))])
    blocks = _chronological_panels(market_results, "chronological_block")
    halves = _chronological_panels(market_results, "chronological_half")
    ordering = _score_ordering(market_results)
    stresses = _stress_panels(
        market_results,
        multipliers=protocol["cost_stress"]["multipliers"],
    )
    power = _prospective_power(candidate, delta, protocol=protocol)
    gate_results = {
        "absolute_market_bootstrap_97_5pct_lcb_gt_zero": (
            candidate_interval["lower"] > 0.0
        ),
        "paired_delta_market_bootstrap_97_5pct_lcb_gt_zero": (
            delta_interval["lower"] > 0.0
        ),
        "every_chronological_block_candidate_total_gte_zero": all(
            panel["candidate_total_unit_net_pnl"] >= 0.0 for panel in blocks.values()
        ),
        "every_chronological_block_paired_delta_total_gte_zero": all(
            panel["paired_delta_total_unit_net_pnl"] >= 0.0 for panel in blocks.values()
        ),
        "largest_winner_removed_candidate_total_gte_zero": candidate_lwr >= 0.0,
        "largest_positive_delta_removed_total_gte_zero": delta_lwr >= 0.0,
        "stable_score_to_realized_pnl_ordering": ordering["passed"],
        "all_cost_stress_candidate_totals_gte_zero": all(
            panel["candidate_total_unit_net_pnl"] >= 0.0 for panel in stresses.values()
        ),
        "all_cost_stress_paired_delta_totals_gte_zero": all(
            panel["paired_delta_total_unit_net_pnl"] >= 0.0 for panel in stresses.values()
        ),
        "prospective_power_required_market_count_lte_2000": (
            power["required_market_count"] is not None
            and power["required_market_count"]
            <= int(protocol["prospective_power"]["maximum_market_count"])
        ),
        "population_and_leakage_reconciliation": (
            len(market_results) == int(protocol["rolling_origin"]["oof_market_count"])
            and len(fold_audits) == int(protocol["rolling_origin"]["target_block_count"])
            and all(
                audit["target_or_future_label_leakage_count"] == 0
                for audit in fold_audits
            )
        ),
    }
    failed = [name for name, passed in gate_results.items() if not passed]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "slot_id": protocol["slot_id"],
        "candidate_role": protocol["candidate_role"],
        "created_at": protocol["created_at"],
        "source_commit": source_commit,
        "protocol_sha256": protocol_sha256,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "promotion_claim_made": False,
        "population": {
            "source_market_count": int(protocol["dataset"]["market_count"]),
            "initial_training_market_count": int(
                protocol["rolling_origin"]["initial_training_market_count"]
            ),
            "oof_market_count": len(market_results),
            "candidate_market_count": len(candidate),
            "baseline_market_count": len(baseline),
            "paired_market_count": len(delta),
            "NO_TRADE_included_as_zero": True,
        },
        "overall": {
            "candidate_accepted_market_count": sum(
                bool(row["candidate_accepted"]) for row in market_results
            ),
            "baseline_accepted_market_count": sum(
                bool(row["baseline_accepted"]) for row in market_results
            ),
            "candidate_total_unit_net_pnl": float(candidate.sum()),
            "baseline_total_unit_net_pnl": float(baseline.sum()),
            "paired_delta_total_unit_net_pnl": float(delta.sum()),
            "candidate_mean_unit_net_pnl": float(candidate.mean()),
            "baseline_mean_unit_net_pnl": float(baseline.mean()),
            "paired_delta_mean_unit_net_pnl": float(delta.mean()),
            "candidate_bootstrap_interval": candidate_interval,
            "baseline_bootstrap_interval": baseline_interval,
            "paired_delta_bootstrap_interval": delta_interval,
            "shared_bootstrap_indices_sha256": hashlib.sha256(
                indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
        "robustness": {
            "candidate_largest_winner_removed_total_unit_net_pnl": candidate_lwr,
            "paired_delta_largest_positive_removed_total_unit_net_pnl": delta_lwr,
            "chronological_blocks": blocks,
            "chronological_halves": halves,
            "score_to_realized_pnl_ordering": ordering,
            "cost_stress": stresses,
        },
        "prospective_power": power,
        "gate_results": gate_results,
        "all_gates_passed": not failed,
        "failed_gates": failed,
        "candidate_freeze_allowed": not failed,
        "live_shadow_start_allowed": False,
        "fresh_confirmatory_collection_authorized": False,
        "threshold_or_hyperparameter_search_performed": False,
        "route_side_missingness_or_outlier_filtering_performed": False,
        "safety": dict(SAFETY),
    }


def render_residual_oof_markdown(report: Mapping[str, Any]) -> str:
    """Render the deterministic human-readable development report."""

    overall = report["overall"]
    power = report["prospective_power"]
    lines = [
        "# BTC 15m cost-aware residual primary slot 001",
        "",
        f"- All OOF gates passed: `{report['all_gates_passed']}`",
        f"- OOF markets: `{report['population']['oof_market_count']}`",
        f"- Candidate accepted markets: `{overall['candidate_accepted_market_count']}`",
        f"- Candidate total unit PnL: `{overall['candidate_total_unit_net_pnl']:.8f}`",
        f"- Matched baseline total unit PnL: `{overall['baseline_total_unit_net_pnl']:.8f}`",
        f"- Paired delta total: `{overall['paired_delta_total_unit_net_pnl']:.8f}`",
        f"- Candidate 97.5% LCB: `{overall['candidate_bootstrap_interval']['lower']:.8f}`",
        f"- Paired-delta 97.5% LCB: `{overall['paired_delta_bootstrap_interval']['lower']:.8f}`",
        f"- Conservative prospective required N: `{power['required_market_count']}`",
        "",
        "## Frozen gates",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`" for name, passed in report["gate_results"].items()
    )
    lines.extend(
        [
            "",
            "This is outcome-aware development evidence only. It is permanently ineligible "
            "for promotion evidence and does not authorize live shadow, fresh collection, "
            "paper/live execution, wallet signing, writes, or capital risk.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_frozen_development_rows(
    *,
    protocol: Mapping[str, Any],
    repository_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    inputs = dict(protocol["inputs"])
    scored_rows = _load_jsonl(
        _verify_descriptor(inputs["terminal_diagnostic_scored_rows"], repository_root=repository_root)
    )
    score_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["side"])): row
        for row in scored_rows
    }
    evaluation_rows = _load_jsonl(
        _verify_descriptor(inputs["confirmatory_market_evaluation_rows"], repository_root=repository_root)
    )
    baseline_by_market = {str(row["market_id"]): row for row in evaluation_rows}
    artifacts = _evaluation_artifacts(
        freeze=PARENT_CONFIG_DIR / "confirmatory_collection_freeze_001",
        config=PARENT_CONFIG_DIR,
    )
    contexts = _load_exact_contexts(
        repository_root=repository_root,
        artifacts=artifacts,
    )
    population_order = [str(context["market_id"]) for context in contexts]
    if len(contexts) != int(protocol["dataset"]["market_count"]):
        raise ValueError("residual source market population changed")
    if set(population_order) != set(baseline_by_market):
        raise ValueError("matched baseline population changed")
    rows = []
    for market_position, context in enumerate(contexts, start=1):
        market_id = str(context["market_id"])
        for feature_row in sorted(
            context["feature_rows"], key=lambda row: int(row["decision_ts"])
        ):
            decision_ts = int(feature_row["decision_ts"])
            for side in SIDES:
                key = (market_id, decision_ts, side)
                scored = score_by_key.get(key)
                if scored is None:
                    raise ValueError("terminal scored target row missing")
                transformed = side_symmetric_features(feature_row, side)
                if tuple(transformed) != FEATURE_NAMES:
                    raise ValueError("108-feature order changed")
                values = np.asarray(
                    [float(transformed[name]) for name in FEATURE_NAMES], dtype=float
                )
                if any(
                    (not math.isfinite(values[index]))
                    and values[index + len(BASE_FEATURE_NAMES)] != 1.0
                    for index in range(len(BASE_FEATURE_NAMES))
                ):
                    raise ValueError("native missing value lacks explicit indicator")
                rows.append(
                    {
                        "market_id": market_id,
                        "market_position": market_position,
                        "market_start_ts": int(context["market"]["market_start_ts"]),
                        "decision_ts": decision_ts,
                        "side": side,
                        "features": values,
                        "feature_row_sha256": scored["feature_row_sha256"],
                        "target": float(scored["realized_unit_net_pnl_if_action"]),
                        "resolved_outcome": scored["resolved_outcome"],
                        "cost_decomposition": dict(scored["cost_decomposition"]),
                    }
                )
    if len(rows) != int(protocol["dataset"]["side_decision_row_count"]):
        raise ValueError("residual side-decision row count changed")
    return rows, baseline_by_market, population_order


def _rolling_origin_predict(
    *,
    rows: Sequence[Mapping[str, Any]],
    population_order: Sequence[str],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_market[str(row["market_id"])].append(row)
    rolling = dict(protocol["rolling_origin"])
    initial = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    block_count = int(rolling["target_block_count"])
    parameters = dict(protocol["model"]["parameters"])
    boost_rounds = int(protocol["model"]["fixed_num_boost_round"])
    predictions = []
    audits = []
    for block_index in range(block_count):
        target_start = initial + block_index * block_size
        target_end = target_start + block_size
        training_ids = list(population_order[:target_start])
        target_ids = list(population_order[target_start:target_end])
        if len(target_ids) != block_size:
            raise ValueError("rolling-origin target block population mismatch")
        train_rows = [row for market_id in training_ids for row in rows_by_market[market_id]]
        target_rows = [row for market_id in target_ids for row in rows_by_market[market_id]]
        train_matrix = _dmatrix(train_rows, with_labels=True)
        target_matrix = _dmatrix(target_rows, with_labels=False)
        booster = xgb.train(
            params=parameters,
            dtrain=train_matrix,
            num_boost_round=boost_rounds,
            verbose_eval=False,
        )
        values = booster.predict(target_matrix)
        for row, value in zip(target_rows, values, strict=True):
            if not math.isfinite(float(value)):
                raise ValueError("residual OOF prediction is not finite")
            predictions.append(
                {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "lineage_id": LINEAGE_ID,
                    "slot_id": protocol["slot_id"],
                    "market_id": row["market_id"],
                    "market_start_ts": row["market_start_ts"],
                    "decision_ts": row["decision_ts"],
                    "side": row["side"],
                    "prediction": float(value),
                    "realized_unit_net_pnl_if_action": row["target"],
                    "resolved_outcome": row["resolved_outcome"],
                    "cost_decomposition": row["cost_decomposition"],
                    "feature_row_sha256": row["feature_row_sha256"],
                    "chronological_block": block_index + 1,
                    "strictly_prior_training_market_count": len(training_ids),
                    "target_or_future_label_used_for_fit": False,
                    "development_only_forever": True,
                    "promotion_evidence_eligible": False,
                    "safety": dict(SAFETY),
                }
            )
        audits.append(
            {
                "schema_version": FOLD_SCHEMA_VERSION,
                "lineage_id": LINEAGE_ID,
                "slot_id": protocol["slot_id"],
                "chronological_block": block_index + 1,
                "strictly_prior_training_market_count": len(training_ids),
                "target_market_count": len(target_ids),
                "training_market_ids_sha256": canonical_json_sha256(training_ids),
                "target_market_ids_sha256": canonical_json_sha256(target_ids),
                "last_training_market_position": target_start,
                "first_target_market_position": target_start + 1,
                "target_or_future_label_leakage_count": 0,
                "fixed_num_boost_round": boost_rounds,
                "model_parameters_sha256": canonical_json_sha256(parameters),
                "development_only_forever": True,
                "promotion_evidence_eligible": False,
                "safety": dict(SAFETY),
            }
        )
    return predictions, audits


def _dmatrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    with_labels: bool,
) -> xgb.DMatrix:
    values = np.vstack([np.asarray(row["features"], dtype=float) for row in rows])
    labels = (
        np.asarray([float(row["target"]) for row in rows], dtype=float)
        if with_labels
        else None
    )
    return xgb.DMatrix(
        values,
        label=labels,
        feature_names=list(FEATURE_NAMES),
        missing=np.nan,
    )


def _public_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(row["features"], dtype=float)
    features = {
        name: (float(value) if math.isfinite(float(value)) else None)
        for name, value in zip(FEATURE_NAMES, values, strict=True)
    }
    return {
        "schema_version": "bigan-btc-15m-cost-aware-residual-dataset-row-v1",
        "lineage_id": LINEAGE_ID,
        "market_id": row["market_id"],
        "market_position": row["market_position"],
        "market_start_ts": row["market_start_ts"],
        "decision_ts": row["decision_ts"],
        "side": row["side"],
        "features": features,
        "feature_row_sha256": row["feature_row_sha256"],
        "target": row["target"],
        "resolved_outcome": row["resolved_outcome"],
        "cost_decomposition": row["cost_decomposition"],
        "native_missing_value": "nan_in_memory_null_on_disk",
        "missing_values_encoded_as_zero": False,
        "development_only_forever": True,
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }


def _bootstrap_indices(*, market_count: int, resamples: int, seed: int) -> np.ndarray:
    if market_count <= 0 or resamples <= 0:
        raise ValueError("bootstrap population and resamples must be positive")
    return np.random.default_rng(seed).integers(
        0, market_count, size=(resamples, market_count), endpoint=False
    )


def _bootstrap_interval(
    values: np.ndarray,
    *,
    indices: np.ndarray,
    confidence: float,
) -> dict[str, Any]:
    means = np.mean(values[indices], axis=1)
    lower_q = 1.0 - confidence
    return {
        "method": "market_level_paired_percentile_bootstrap",
        "confidence": confidence,
        "resamples": int(indices.shape[0]),
        "lower": float(np.quantile(means, lower_q)),
        "upper": float(np.quantile(means, confidence)),
    }


def _chronological_panels(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: {
            "market_count": len(group),
            "candidate_accepted_market_count": sum(
                bool(row["candidate_accepted"]) for row in group
            ),
            "candidate_total_unit_net_pnl": float(
                sum(float(row["candidate_unit_net_pnl"]) for row in group)
            ),
            "baseline_total_unit_net_pnl": float(
                sum(float(row["baseline_unit_net_pnl"]) for row in group)
            ),
            "paired_delta_total_unit_net_pnl": float(
                sum(float(row["paired_delta_unit_net_pnl"]) for row in group)
            ),
        }
        for name, group in sorted(groups.items(), key=lambda item: item[0])
    }


def _score_ordering(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["candidate_accepted"]]
    if len(accepted) < 4:
        return {
            "accepted_market_count": len(accepted),
            "spearman_rank_correlation": None,
            "high_score_half_mean_pnl": None,
            "low_score_half_mean_pnl": None,
            "high_minus_low_mean_pnl": None,
            "passed": False,
        }
    scores = np.asarray([float(row["candidate_prediction"]) for row in accepted])
    pnl = np.asarray([float(row["candidate_unit_net_pnl"]) for row in accepted])
    correlation = _spearman(scores, pnl)
    ordered = np.argsort(scores, kind="stable")
    midpoint = len(ordered) // 2
    low = float(pnl[ordered[:midpoint]].mean())
    high = float(pnl[ordered[midpoint:]].mean())
    return {
        "accepted_market_count": len(accepted),
        "spearman_rank_correlation": correlation,
        "high_score_half_mean_pnl": high,
        "low_score_half_mean_pnl": low,
        "high_minus_low_mean_pnl": high - low,
        "passed": correlation > 0.0 and high >= low,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _stress_panels(
    rows: Sequence[Mapping[str, Any]],
    *,
    multipliers: Sequence[float],
) -> dict[str, dict[str, Any]]:
    output = {}
    for multiplier in multipliers:
        candidate = np.asarray(
            [
                float(row["candidate_unit_net_pnl"])
                - (float(multiplier) - 1.0)
                * float(row["candidate_total_cost_relative_to_mid"])
                for row in rows
            ]
        )
        baseline = np.asarray(
            [
                float(row["baseline_unit_net_pnl"])
                - (float(multiplier) - 1.0)
                * float(row["baseline_total_cost_relative_to_mid"])
                for row in rows
            ]
        )
        output[f"{float(multiplier):.1f}x"] = {
            "multiplier": float(multiplier),
            "candidate_total_unit_net_pnl": float(candidate.sum()),
            "baseline_total_unit_net_pnl": float(baseline.sum()),
            "paired_delta_total_unit_net_pnl": float((candidate - baseline).sum()),
            "actions_reselected": False,
        }
    return output


def _prospective_power(
    candidate: np.ndarray,
    delta: np.ndarray,
    *,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    power = dict(protocol["prospective_power"])
    confidence = float(power["confidence"])
    target_power = float(power["target_power"])
    haircut = float(power["effect_haircut"])
    z_gate = NormalDist().inv_cdf(confidence)
    z_power = NormalDist().inv_cdf(target_power)
    panels = {}
    required = []
    for name, values in (("absolute_candidate", candidate), ("paired_delta", delta)):
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        conservative_effect = mean * haircut
        required_n = (
            math.ceil(
                ((z_gate + z_power) * standard_deviation / conservative_effect) ** 2
            )
            if conservative_effect > 0.0 and standard_deviation > 0.0
            else None
        )
        panels[name] = {
            "observed_mean": mean,
            "observed_standard_deviation": standard_deviation,
            "effect_haircut": haircut,
            "conservative_effect": conservative_effect,
            "required_market_count": required_n,
        }
        if required_n is not None:
            required.append(required_n)
    required_market_count = max(required) if len(required) == 2 else None
    return {
        "method": "plugin_normal_approximation_with_preregistered_effect_haircut",
        "confidence": confidence,
        "target_power": target_power,
        "maximum_market_count": int(power["maximum_market_count"]),
        "panels": panels,
        "required_market_count": required_market_count,
        "fast_track_feasible": (
            required_market_count is not None
            and required_market_count <= int(power["maximum_market_count"])
        ),
        "diagnostic_only_not_promotion_evidence": True,
    }


def _stress_cost_from_prediction(row: Mapping[str, Any]) -> float:
    cost = dict(row["cost_decomposition"])
    spread = (float(cost["entry_ask"]) - float(cost["entry_bid"])) / 2.0
    total = spread + float(cost["fees"]) + float(cost["slippage"]) + float(
        cost["liquidity_impact"]
    )
    if total < 0.0 or not math.isfinite(total):
        raise ValueError("candidate stress cost is invalid")
    return total


def _validate_market_results(
    rows: Sequence[Mapping[str, Any]], *, protocol: Mapping[str, Any]
) -> None:
    expected = int(protocol["rolling_origin"]["oof_market_count"])
    if len(rows) != expected or len({str(row["market_id"]) for row in rows}) != expected:
        raise ValueError("residual OOF market population mismatch")
    if [int(row["oof_position"]) for row in rows] != list(range(1, expected + 1)):
        raise ValueError("residual OOF market order changed")
    if any(
        row.get("schema_version") != MARKET_RESULT_SCHEMA_VERSION
        or row.get("development_only_forever") is not True
        or row.get("promotion_evidence_eligible") is not False
        or dict(row.get("safety") or {}) != SAFETY
        or abs(
            float(row["candidate_unit_net_pnl"])
            - float(row["baseline_unit_net_pnl"])
            - float(row["paired_delta_unit_net_pnl"])
        )
        > 1e-12
        for row in rows
    ):
        raise ValueError("residual OOF market row reconciliation failed")


def _validate_fold_audits(
    rows: Sequence[Mapping[str, Any]], *, protocol: Mapping[str, Any]
) -> None:
    rolling = dict(protocol["rolling_origin"])
    block_count = int(rolling["target_block_count"])
    if len(rows) != block_count:
        raise ValueError("residual OOF fold count mismatch")
    expected_training = int(rolling["initial_training_market_count"])
    block_size = int(rolling["target_block_size"])
    for ordinal, row in enumerate(rows, start=1):
        if not (
            row.get("schema_version") == FOLD_SCHEMA_VERSION
            and row.get("chronological_block") == ordinal
            and row.get("strictly_prior_training_market_count") == expected_training
            and row.get("target_market_count") == block_size
            and row.get("last_training_market_position")
            < row.get("first_target_market_position")
            and row.get("target_or_future_label_leakage_count") == 0
            and dict(row.get("safety") or {}) == SAFETY
        ):
            raise ValueError("residual OOF fold audit mismatch")
        expected_training += block_size


def _validate_frozen_population(
    *,
    predictions: Sequence[Mapping[str, Any]],
    fold_audits: Sequence[Mapping[str, Any]],
    market_results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> None:
    _validate_fold_audits(fold_audits, protocol=protocol)
    _validate_market_results(market_results, protocol=protocol)
    expected_predictions = int(protocol["rolling_origin"]["oof_market_count"]) * 4
    if len(predictions) != expected_predictions:
        raise ValueError("residual OOF prediction count mismatch")
    prediction_markets = {str(row["market_id"]) for row in predictions}
    result_markets = {str(row["market_id"]) for row in market_results}
    if prediction_markets != result_markets:
        raise ValueError("residual OOF prediction/result population mismatch")
    if any(
        row.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or row.get("target_or_future_label_used_for_fit") is not False
        or dict(row.get("safety") or {}) != SAFETY
        for row in predictions
    ):
        raise ValueError("residual OOF prediction governance mismatch")


def _verify_descriptor(
    descriptor: Mapping[str, Any], *, repository_root: Path
) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("residual artifact descriptor field mismatch")
    path = Path(str(descriptor["path"]))
    if path.is_absolute():
        raise ValueError("machine-local residual artifact path forbidden")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError("residual artifact unavailable")
    if sha256_file(resolved) != str(descriptor["sha256"]):
        raise ValueError("residual artifact SHA mismatch")
    return resolved


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("residual artifact escaped repository")
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"frozen residual JSON unavailable: {path}")
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"frozen residual JSON sidecar mismatch: {path}")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _looks_like_git_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


__all__ = [
    "build_residual_oof_report",
    "market_results_from_predictions",
    "render_residual_oof_markdown",
    "run_residual_rolling_origin_oof",
    "validate_residual_oof_protocol",
    "verify_frozen_residual_oof",
]
