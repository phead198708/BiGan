"""One-shot confirmatory evaluator for residual promotion v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.cost_aware_residual import (
    _bootstrap_indices,
    _bootstrap_interval,
    _chronological_panels,
    _score_ordering,
    _stress_panels,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.residual_promotion_finalization import (
    validate_frozen_population,
)
from bigan.v8.polymarket.residual_promotion_v1 import (
    CANDIDATE_ID,
    LINEAGE_ID,
    TARGET_MARKETS,
)

EVALUATION_SCHEMA_VERSION = "bigan-btc-15m-residual-promotion-evaluation-v1"
MARKET_RESULT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-market-result-v1"
)
EXECUTION_CONTRACT_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-evaluation-execution-contract-v1"
)
AUTHORIZATION_SCHEMA_VERSION = (
    "bigan-btc-15m-residual-promotion-outcome-evaluation-authorization-v1"
)
IMPLEMENTATION_REPOSITORY_PATH = (
    "src/bigan/v8/polymarket/residual_promotion_evaluation.py"
)
CONFIG_REPOSITORY_PATH = (
    "examples/v8/polymarket_configs/"
    "BTC-15M-cost-aware-market-residual-promotion-v1"
)
REQUIRED_GATE_NAMES = {
    "absolute_candidate_bootstrap_97_5pct_lcb_gt_zero",
    "paired_delta_bootstrap_97_5pct_lcb_gt_zero",
    "both_chronological_halves_candidate_total_gte_zero",
    "both_chronological_halves_delta_total_gte_zero",
    "every_one_of_five_chronological_500_market_blocks_candidate_total_gte_zero",
    "every_one_of_five_chronological_500_market_blocks_delta_total_gte_zero",
    "largest_positive_delta_removed_total_gte_zero",
    "largest_winner_removed_candidate_total_gte_zero",
    "cost_stress_1_2_1_5_2x_candidate_and_delta_totals_gte_zero",
    "market_identity_and_population_reconciliation",
    "missingness_and_causality_reconciliation",
    "offline_live_prediction_and_decision_parity",
    "stable_score_to_realized_pnl_ordering",
}


def run_authorized_promotion_evaluation(
    *,
    repository_root: Path | str,
    service_root: Path | str,
    freeze_dir: Path | str,
    expected_population_manifest_sha256: str,
    settlements_path: Path | str,
    expected_settlements_sha256: str,
    execution_contract_path: Path | str,
    expected_execution_contract_sha256: str,
    authorization_path: Path | str,
    expected_authorization_sha256: str,
    output_dir: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Run the frozen confirmatory evaluation exactly once after authorization."""

    root = Path(repository_root).resolve()
    collection_root = Path(service_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("promotion evaluation output already exists; rerun forbidden")
    contract_path = _repo_file(execution_contract_path, root)
    if sha256_file(contract_path) != expected_execution_contract_sha256:
        raise ValueError("evaluation execution contract SHA-256 mismatch")
    contract = _verified_json(contract_path)
    validate_evaluation_execution_contract(contract, repository_root=root)
    authorization_file = _repo_file(authorization_path, root)
    if sha256_file(authorization_file) != expected_authorization_sha256:
        raise ValueError("outcome evaluation authorization SHA-256 mismatch")
    authorization = _verified_json(authorization_file)
    _validate_evaluation_authorization(
        authorization,
        execution_contract=_descriptor(contract_path, root),
        population_manifest_sha256=expected_population_manifest_sha256,
    )
    stamp = created_at or datetime.now(UTC).isoformat()
    destination.mkdir(parents=True, exist_ok=False)
    start_descriptor = _write_json(
        destination / "promotion_evaluation_started.json",
        {
            "schema_version": (
                "bigan-btc-15m-residual-promotion-evaluation-start-v1"
            ),
            "lineage_id": LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "started_at": stamp,
            "execution_contract": _descriptor(contract_path, root),
            "evaluation_authorization": _descriptor(authorization_file, root),
            "population_manifest_sha256": expected_population_manifest_sha256,
            "expected_official_settlements_sha256": expected_settlements_sha256,
            "evaluation_slot_consumed": True,
            "rerun_allowed": False,
            "automatic_promotion_or_live_unlock": False,
            "safety": dict(SAFETY),
        },
    )
    freeze_validation = validate_frozen_population(
        freeze_dir=freeze_dir,
        service_root=collection_root,
        repository_root=root,
        expected_manifest_sha256=expected_population_manifest_sha256,
    )
    if freeze_validation["validation_passed"] is not True:
        raise ValueError("exact population validation did not pass")
    freeze = Path(freeze_dir).resolve()
    if not freeze.is_relative_to(collection_root):
        raise ValueError("exact population freeze escaped collection service root")
    candidate_rows = _load_jsonl(freeze / "candidate_decision_rows.jsonl")
    baseline_rows = _load_jsonl(freeze / "baseline_decision_rows.jsonl")
    settlement_file = Path(settlements_path).resolve()
    if (
        not settlement_file.is_relative_to(collection_root)
        or not settlement_file.is_file()
        or sha256_file(settlement_file) != expected_settlements_sha256
    ):
        raise ValueError("official settlement artifact SHA-256 mismatch")
    settlements = _load_jsonl(settlement_file)
    market_results, reconciliation = build_market_results(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        settlements=settlements,
        target_market_count=TARGET_MARKETS,
    )
    protocol_descriptor = dict(contract["statistical_protocol"])
    protocol = _verified_json(root / protocol_descriptor["path"])
    parity_descriptor = dict(contract["runtime_parity_report"])
    parity = _verified_json(root / parity_descriptor["path"])
    report = build_promotion_report(
        market_results=market_results,
        protocol=protocol,
        reconciliation=reconciliation,
        runtime_parity_passed=parity.get("prediction_and_decision_parity") is True,
        production=True,
        created_at=stamp,
    )
    results_descriptor = _write_jsonl(
        destination / "promotion_market_results.jsonl", market_results
    )
    report_descriptor = _write_json(
        destination / "promotion_evaluation_report.json", report
    )
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": report["created_at"],
        "execution_contract": _descriptor(contract_path, root),
        "evaluation_authorization": _descriptor(authorization_file, root),
        "population_manifest_sha256": expected_population_manifest_sha256,
        "official_settlements": {
            "path": settlement_file.relative_to(collection_root).as_posix(),
            "sha256": expected_settlements_sha256,
        },
        "evaluation_start": start_descriptor,
        "market_results": results_descriptor,
        "evaluation_report": report_descriptor,
        "evaluation_executed_exactly_once": True,
        "rerun_allowed": False,
        "fresh_population_reuse_allowed": False,
        "all_fresh_confirmation_gates_passed": report["all_gates_passed"],
        "lineage_terminalized": report["lineage_terminalized"],
        "automatic_promotion_or_live_unlock": False,
        "micro_live_approval_granted": False,
        "safety": dict(SAFETY),
    }
    manifest_descriptor = _write_json(
        destination / "promotion_evaluation_manifest.json", manifest
    )
    return {
        "manifest": manifest_descriptor,
        "report": report_descriptor,
        "all_gates_passed": report["all_gates_passed"],
        "lineage_terminalized": report["lineage_terminalized"],
        "automatic_promotion_or_live_unlock": False,
        "safety": dict(SAFETY),
    }


def validate_evaluation_execution_contract(
    contract: Mapping[str, Any], *, repository_root: Path | str
) -> None:
    """Verify the pre-outcome execution contract and all repository bindings."""

    root = Path(repository_root).resolve()
    if not (
        contract.get("schema_version") == EXECUTION_CONTRACT_SCHEMA_VERSION
        and contract.get("lineage_id") == LINEAGE_ID
        and contract.get("candidate_id") == CANDIDATE_ID
        and contract.get("target_quality_valid_market_count") == TARGET_MARKETS
        and contract.get("market_level_rows_required") is True
        and contract.get("evaluation_exactly_once") is True
        and contract.get("rerun_allowed") is False
        and contract.get("interim_evaluation_allowed") is False
        and contract.get("optional_stopping_allowed") is False
        and contract.get("outcomes_accessed_when_frozen") is False
        and contract.get("evaluation_authorized_by_contract") is False
        and contract.get("automatic_promotion_or_live_unlock") is False
        and set(contract.get("required_gate_names") or []) == REQUIRED_GATE_NAMES
        and dict(contract.get("safety") or {}) == SAFETY
    ):
        raise ValueError("promotion evaluation execution contract is invalid")
    bound = {
        "implementation",
        "finalization_implementation",
        "statistical_protocol",
        "reporting_contract",
        "candidate_bundle",
        "baseline_artifact",
        "cost_contract",
        "feature_contract",
        "gate_implementation",
        "runtime_parity_report",
    }
    revision = contract.get("contract_revision")
    if revision == "native_missingness_reconciliation_v2":
        bound.update({"finalization_correction", "supersedes_execution_contract"})
    elif revision == "feature_envelope_reconciliation_v3":
        bound.update(
            {
                "finalization_correction",
                "finalization_feature_envelope_correction",
                "supersedes_execution_contract",
            }
        )
    elif revision is not None:
        raise ValueError("unknown promotion evaluation contract revision")
    for name in bound:
        _verify_descriptor(dict(contract.get(name) or {}), repository_root=root)
    if dict(contract["implementation"])["path"] != IMPLEMENTATION_REPOSITORY_PATH:
        raise ValueError("promotion evaluator implementation path mismatch")
    protocol = _verified_json(root / dict(contract["statistical_protocol"])["path"])
    if (
        set(dict(protocol.get("gates") or {})) != REQUIRED_GATE_NAMES
        or any(value is not True for value in dict(protocol["gates"]).values())
        or protocol.get("fresh_outcomes_accessed") is not False
        or dict(protocol.get("safety") or {}) != SAFETY
    ):
        raise ValueError("frozen statistical gate contract mismatch")
    for name in (
        "candidate_bundle",
        "baseline_artifact",
        "cost_contract",
        "feature_contract",
        "runtime_parity_report",
    ):
        if _pair(protocol.get(name)) != _pair(contract.get(name)):
            raise ValueError(f"execution/statistical contract binding mismatch: {name}")
    bootstrap = dict(protocol.get("bootstrap") or {})
    population = dict(protocol.get("population") or {})
    if not (
        bootstrap.get("method") == "market_level_paired_percentile_bootstrap"
        and bootstrap.get("confidence") == 0.975
        and bootstrap.get("lower_quantile") == 0.025
        and bootstrap.get("resamples") == 10000
        and bootstrap.get("candidate_and_baseline_share_indices") is True
        and bootstrap.get("NO_TRADE_participates_as_zero") is True
        and population.get("target_quality_valid_market_count") == TARGET_MARKETS
        and population.get("maximum_attempts") == 3000
    ):
        raise ValueError("frozen bootstrap or population contract mismatch")


def build_market_results(
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    settlements: Sequence[Mapping[str, Any]],
    target_market_count: int,
    synthetic_dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconcile exact ordered populations and compute unit HOLD_TO_SETTLEMENT PnL."""

    if not (
        len(candidate_rows)
        == len(baseline_rows)
        == len(settlements)
        == target_market_count
    ):
        raise ValueError("candidate/baseline/settlement population count mismatch")
    candidate_ids = [str(row.get("market_id") or "") for row in candidate_rows]
    baseline_ids = [str(row.get("market_id") or "") for row in baseline_rows]
    settlement_ids = [str(row.get("market_id") or "") for row in settlements]
    if not (
        candidate_ids == baseline_ids == settlement_ids
        and len(set(candidate_ids)) == target_market_count
    ):
        raise ValueError("candidate/baseline/settlement population identity mismatch")
    block_size = target_market_count // 5
    if target_market_count % 5 != 0 or target_market_count % 2 != 0:
        raise ValueError("evaluation population cannot form frozen blocks and halves")
    results = []
    for index, (candidate, baseline, settlement) in enumerate(
        zip(candidate_rows, baseline_rows, settlements, strict=True), start=1
    ):
        _validate_official_settlement(
            settlement, synthetic_dry_run=synthetic_dry_run
        )
        candidate_result = _policy_result(candidate, settlement)
        baseline_result = _policy_result(baseline, settlement)
        result = {
            "schema_version": MARKET_RESULT_SCHEMA_VERSION,
            "lineage_id": LINEAGE_ID,
            "candidate_id": CANDIDATE_ID,
            "market_id": candidate_ids[index - 1],
            "population_position": index,
            "chronological_block": (index - 1) // block_size + 1,
            "chronological_half": (
                "first" if index <= target_market_count // 2 else "second"
            ),
            "candidate_accepted": candidate_result["accepted"],
            "candidate_selected_side": candidate_result["selected_side"],
            "candidate_decision_ts": candidate_result["decision_ts"],
            "candidate_prediction": candidate_result["selected_action_value"],
            "candidate_unit_net_pnl": candidate_result["unit_net_pnl"],
            "candidate_total_cost_relative_to_mid": candidate_result[
                "total_cost_relative_to_mid"
            ],
            "candidate_cost_decomposition": candidate_result[
                "cost_decomposition"
            ],
            "baseline_accepted": baseline_result["accepted"],
            "baseline_selected_side": baseline_result["selected_side"],
            "baseline_decision_ts": baseline_result["decision_ts"],
            "baseline_unit_net_pnl": baseline_result["unit_net_pnl"],
            "baseline_total_cost_relative_to_mid": baseline_result[
                "total_cost_relative_to_mid"
            ],
            "baseline_cost_decomposition": baseline_result["cost_decomposition"],
            "paired_delta_unit_net_pnl": (
                candidate_result["unit_net_pnl"] - baseline_result["unit_net_pnl"]
            ),
            "official_settlement_source": settlement["settlement_source"],
            "NO_TRADE_participates_as_zero": True,
            "safety": dict(SAFETY),
        }
        results.append(result)
    reconciliation = {
        "target_market_count": target_market_count,
        "candidate_market_count": len(candidate_rows),
        "baseline_market_count": len(baseline_rows),
        "official_settlement_market_count": len(settlements),
        "paired_market_count": len(results),
        "ordered_market_ids_sha256": canonical_json_sha256(candidate_ids),
        "candidate_baseline_settlement_order_equal": True,
        "duplicate_market_count": 0,
        "unresolved_market_count": 0,
        "inferred_settlement_count": 0,
        "missing_execution_feature_count": 0,
        "causality_violation_count": 0,
        "passed": True,
    }
    return results, reconciliation


def build_promotion_report(
    *,
    market_results: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    runtime_parity_passed: bool,
    production: bool,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Apply the unchanged frozen promotion gates to one exact population."""

    market_count = len(market_results)
    if production and market_count != TARGET_MARKETS:
        raise ValueError("production promotion evaluation requires exactly 2500 markets")
    if not market_results:
        raise ValueError("promotion evaluation population is empty")
    candidate = np.asarray(
        [float(row["candidate_unit_net_pnl"]) for row in market_results]
    )
    baseline = np.asarray(
        [float(row["baseline_unit_net_pnl"]) for row in market_results]
    )
    delta = candidate - baseline
    bootstrap = dict(protocol["bootstrap"])
    indices = _bootstrap_indices(
        market_count=market_count,
        resamples=int(bootstrap["resamples"]),
        seed=int(bootstrap["seed"]),
    )
    confidence = float(bootstrap["confidence"])
    candidate_interval = _bootstrap_interval(
        candidate, indices=indices, confidence=confidence
    )
    baseline_interval = _bootstrap_interval(
        baseline, indices=indices, confidence=confidence
    )
    delta_interval = _bootstrap_interval(delta, indices=indices, confidence=confidence)
    candidate_lwr = float(candidate.sum() - candidate[int(np.argmax(candidate))])
    delta_lwr = float(delta.sum() - delta[int(np.argmax(delta))])
    blocks = _chronological_panels(market_results, "chronological_block")
    halves = _chronological_panels(market_results, "chronological_half")
    ordering = _score_ordering(market_results)
    stresses = _stress_panels(
        market_results,
        multipliers=list(protocol["cost_stress_multipliers"]),
    )
    gate_results = {
        "absolute_candidate_bootstrap_97_5pct_lcb_gt_zero": (
            candidate_interval["lower"] > 0.0
        ),
        "paired_delta_bootstrap_97_5pct_lcb_gt_zero": delta_interval["lower"] > 0.0,
        "both_chronological_halves_candidate_total_gte_zero": all(
            panel["candidate_total_unit_net_pnl"] >= 0.0
            for panel in halves.values()
        ),
        "both_chronological_halves_delta_total_gte_zero": all(
            panel["paired_delta_total_unit_net_pnl"] >= 0.0
            for panel in halves.values()
        ),
        "every_one_of_five_chronological_500_market_blocks_candidate_total_gte_zero": (
            len(blocks) == 5
            and all(
                panel["candidate_total_unit_net_pnl"] >= 0.0
                for panel in blocks.values()
            )
        ),
        "every_one_of_five_chronological_500_market_blocks_delta_total_gte_zero": (
            len(blocks) == 5
            and all(
                panel["paired_delta_total_unit_net_pnl"] >= 0.0
                for panel in blocks.values()
            )
        ),
        "largest_positive_delta_removed_total_gte_zero": delta_lwr >= 0.0,
        "largest_winner_removed_candidate_total_gte_zero": candidate_lwr >= 0.0,
        "cost_stress_1_2_1_5_2x_candidate_and_delta_totals_gte_zero": all(
            panel["candidate_total_unit_net_pnl"] >= 0.0
            and panel["paired_delta_total_unit_net_pnl"] >= 0.0
            for panel in stresses.values()
        ),
        "market_identity_and_population_reconciliation": (
            reconciliation.get("passed") is True
            and reconciliation.get("paired_market_count") == market_count
        ),
        "missingness_and_causality_reconciliation": (
            reconciliation.get("missing_execution_feature_count") == 0
            and reconciliation.get("causality_violation_count") == 0
        ),
        "offline_live_prediction_and_decision_parity": runtime_parity_passed,
        "stable_score_to_realized_pnl_ordering": ordering["passed"],
    }
    if set(gate_results) != REQUIRED_GATE_NAMES:
        raise ValueError("promotion gate implementation is incomplete")
    failed = sorted(name for name, passed in gate_results.items() if not passed)
    all_passed = not failed
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "lineage_id": LINEAGE_ID,
        "candidate_id": CANDIDATE_ID,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "production_evaluation": production,
        "population": dict(reconciliation),
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
        "gate_results": gate_results,
        "all_gates_passed": all_passed,
        "failed_gates": failed,
        "lineage_terminalized": not all_passed,
        "failed_population_reuse_allowed": False,
        "phase6_required": all_passed,
        "rollback_drill_required": all_passed,
        "micro_live_go_no_go": (
            "NO_GO_PENDING_PHASE6_AND_ROLLBACK_DRILL"
            if all_passed
            else "NO_GO_LINEAGE_TERMINALIZED"
        ),
        "automatic_promotion_or_live_unlock": False,
        "promotion_evidence_eligible": False,
        "paper_candidate_allowed": False,
        "v8_execution_handoff_allowed": False,
        "live_trading_allowed": False,
        "wallet_signing_allowed": False,
        "polymarket_write_allowed": False,
        "capital_at_risk": False,
        "safety": dict(SAFETY),
    }


def dry_run_evaluation_pipeline(*, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise the exact operators with synthetic labels and emit no gate result."""

    candidate_rows = []
    baseline_rows = []
    settlements = []
    for index in range(10):
        market_id = f"synthetic-{index:02d}"
        common = {
            "market_id": market_id,
            "population_position": index + 1,
            "decision_ts": 2_000_000_000_000 + index,
            "accepted": index % 2 == 0,
            "selected_action": "BUY_UP_HOLD" if index % 2 == 0 else "NO_TRADE",
            "selected_side": "UP" if index % 2 == 0 else None,
            "selected_action_value": 0.05 if index % 2 == 0 else 0.0,
            "execution_features": {
                "up_ask": 0.45,
                "up_bid": 0.43,
                "up_liquidity_depth": 10.0,
                "down_ask": 0.57,
                "down_bid": 0.55,
                "down_liquidity_depth": 10.0,
            },
        }
        common["execution_features_sha256"] = canonical_json_sha256(
            common["execution_features"]
        )
        candidate_rows.append(dict(common))
        baseline_rows.append({**common, "accepted": False, "selected_action": "NO_TRADE", "selected_side": None, "selected_action_value": 0.0})
        settlements.append(
            {
                "market_id": market_id,
                "settlement_source": "synthetic_dry_run_only",
                "official_final": True,
                "inferred": False,
                "unresolved": False,
                "payout_up": 1.0 if index % 2 == 0 else 0.0,
                "payout_down": 0.0 if index % 2 == 0 else 1.0,
            }
        )
    results, reconciliation = build_market_results(
        candidate_rows=candidate_rows,
        baseline_rows=baseline_rows,
        settlements=settlements,
        target_market_count=10,
        synthetic_dry_run=True,
    )
    internal = build_promotion_report(
        market_results=results,
        protocol=protocol,
        reconciliation=reconciliation,
        runtime_parity_passed=True,
        production=False,
        created_at="synthetic-dry-run",
    )
    return {
        "schema_version": "bigan-residual-promotion-evaluation-dry-run-v1",
        "lineage_id": LINEAGE_ID,
        "synthetic_only": True,
        "current_confirmatory_outcomes_accessed": False,
        "current_confirmatory_settlement_accessed": False,
        "current_confirmatory_pnl_accessed": False,
        "population_alignment_passed": reconciliation["passed"],
        "shared_bootstrap_indices_sha256": internal["overall"][
            "shared_bootstrap_indices_sha256"
        ],
        "largest_winner_removal_exercised": True,
        "chronological_halves_exercised": True,
        "five_blocks_exercised": len(internal["robustness"]["chronological_blocks"]) == 5,
        "cost_stress_exercised": True,
        "gate_results_emitted": False,
        "promotion_or_pass_result_emitted": False,
        "automatic_promotion_or_live_unlock": False,
        "safety": dict(SAFETY),
    }


def _policy_result(
    decision: Mapping[str, Any], settlement: Mapping[str, Any]
) -> dict[str, Any]:
    features = dict(decision.get("execution_features") or {})
    if canonical_json_sha256(features) != decision.get("execution_features_sha256"):
        raise ValueError("decision execution feature SHA-256 mismatch")
    accepted = decision.get("accepted") is True
    selected_side = decision.get("selected_side")
    if not accepted:
        if selected_side is not None or decision.get("selected_action") != "NO_TRADE":
            raise ValueError("NO_TRADE decision semantics are invalid")
        return {
            "accepted": False,
            "selected_side": None,
            "decision_ts": int(decision["decision_ts"]),
            "selected_action_value": 0.0,
            "unit_net_pnl": 0.0,
            "total_cost_relative_to_mid": 0.0,
            "cost_decomposition": {
                "gross_price_edge": 0.0,
                "entry_spread_cost": 0.0,
                "fees": 0.0,
                "slippage": 0.0,
                "liquidity_impact": 0.0,
                "total_cost": 0.0,
                "unit_net_pnl": 0.0,
            },
        }
    if selected_side not in {"UP", "DOWN"}:
        raise ValueError("accepted decision selected side is invalid")
    prefix = selected_side.lower()
    ask = float(features[f"{prefix}_ask"])
    bid = float(features[f"{prefix}_bid"])
    depth = float(features[f"{prefix}_liquidity_depth"])
    if not (
        math.isfinite(ask)
        and math.isfinite(bid)
        and math.isfinite(depth)
        and 0.0 < bid <= ask < 1.0
        and depth >= 0.0
    ):
        raise ValueError("accepted decision has invalid executable prices")
    midpoint = (ask + bid) / 2.0
    fee = 0.0002
    slippage = max(0.0001, (ask - bid) / 2.0)
    impact = 0.00005 if depth > 0.0 else 0.001
    spread = ask - midpoint
    payout = float(settlement[f"payout_{prefix}"])
    gross = payout - midpoint
    total_cost = spread + fee + slippage + impact
    net = gross - total_cost
    return {
        "accepted": True,
        "selected_side": selected_side,
        "decision_ts": int(decision["decision_ts"]),
        "selected_action_value": float(decision["selected_action_value"]),
        "unit_net_pnl": net,
        "total_cost_relative_to_mid": total_cost,
        "cost_decomposition": {
            "entry_ask": ask,
            "entry_bid": bid,
            "entry_mid": midpoint,
            "gross_price_edge": gross,
            "entry_spread_cost": spread,
            "fees": fee,
            "slippage": slippage,
            "liquidity_impact": impact,
            "total_cost": total_cost,
            "unit_net_pnl": net,
        },
    }


def _validate_official_settlement(
    row: Mapping[str, Any], *, synthetic_dry_run: bool
) -> None:
    payout_up = row.get("payout_up")
    payout_down = row.get("payout_down")
    if not (
        row.get("official_final") is True
        and row.get("inferred") is False
        and row.get("unresolved") is False
        and row.get("settlement_source")
        == (
            "synthetic_dry_run_only"
            if synthetic_dry_run
            else "official_polymarket"
        )
        and (
            synthetic_dry_run
            or (
                isinstance(row.get("official_resolution_reference"), str)
                and bool(row.get("official_resolution_reference"))
                and isinstance(row.get("settlement_finalized_at"), str)
                and bool(row.get("settlement_finalized_at"))
            )
        )
        and payout_up in {0, 1}
        and payout_down in {0, 1}
        and float(payout_up) + float(payout_down) == 1.0
    ):
        raise ValueError("official settlement row is invalid or unresolved")


def _validate_evaluation_authorization(
    authorization: Mapping[str, Any],
    *,
    execution_contract: Mapping[str, Any],
    population_manifest_sha256: str,
) -> None:
    if not (
        authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
        and authorization.get("lineage_id") == LINEAGE_ID
        and authorization.get("fresh_outcome_access_authorized") is True
        and authorization.get("official_settlement_ingestion_authorized") is True
        and authorization.get("evaluation_exactly_once_authorized") is True
        and authorization.get("interim_evaluation_authorized") is False
        and authorization.get("rerun_authorized") is False
        and authorization.get("population_manifest_sha256")
        == population_manifest_sha256
        and _pair(authorization.get("execution_contract"))
        == _pair(execution_contract)
        and isinstance(authorization.get("service_root_id"), str)
        and bool(authorization.get("service_root_id"))
        and isinstance(authorization.get("collection_start_record_sha256"), str)
        and len(authorization.get("collection_start_record_sha256")) == 64
        and authorization.get("paper_live_wallet_write_or_capital_authorized") is False
        and dict(authorization.get("safety") or {}) == SAFETY
    ):
        raise ValueError("outcome evaluation authorization is invalid")


def _repo_file(path: Path | str, repository_root: Path) -> Path:
    value = Path(path)
    resolved = value.resolve() if value.is_absolute() else (repository_root / value).resolve()
    if not resolved.is_relative_to(repository_root) or not resolved.is_file():
        raise ValueError("repository artifact is unavailable")
    return resolved


def _verify_descriptor(
    descriptor: Mapping[str, Any], *, repository_root: Path
) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("repository artifact descriptor is invalid")
    path = _repo_file(str(descriptor["path"]), repository_root)
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("repository artifact descriptor SHA-256 mismatch")
    return path


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = _repo_file(path, repository_root)
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _pair(value: Any) -> dict[str, str]:
    descriptor = dict(value or {})
    return {
        "path": str(descriptor.get("path") or ""),
        "sha256": str(descriptor.get("sha256") or ""),
    }


def _verified_json(path: Path) -> dict[str, Any]:
    sidecars = [
        candidate
        for candidate in (
            path.with_suffix(".sha256"),
            path.with_suffix(path.suffix + ".sha256"),
        )
        if candidate.is_file()
    ]
    if (
        len(sidecars) != 1
        or sidecars[0].read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise ValueError("frozen JSON sidecar mismatch")
    return _load_json(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    raw = b"".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(raw)
    _write_sidecar(path, raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    raw = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    _write_sidecar(path, raw)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _write_sidecar(path: Path, raw: bytes) -> None:
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8"
    )


__all__ = [
    "build_market_results",
    "build_promotion_report",
    "dry_run_evaluation_pipeline",
    "run_authorized_promotion_evaluation",
    "validate_evaluation_execution_contract",
]
