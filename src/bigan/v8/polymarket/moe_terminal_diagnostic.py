"""Deterministic post-terminal diagnostics for BTC-15M-MoE-confirmatory-v2."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from bigan.v8.polymarket.challenge_development_lane import sha256_file
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.moe_collection_boundary_r2 import (
    _write_new_frozen_json,
    _write_new_jsonl,
)
from bigan.v8.polymarket.moe_collection_observability import (
    _execution_costs,
    _feature_matrix,
    _load_runtime_bundle,
    _pair_normalized_predictions,
    _router_observation,
)
from bigan.v8.polymarket.moe_confirmatory_evaluation import (
    _evaluation_artifacts,
    _load_exact_contexts,
    _write_new_frozen_text,
)
from bigan.v8.polymarket.moe_confirmatory_lineage import (
    deterministic_moe_route,
    frozen_expert_or_fallback,
)
from bigan.v8.polymarket.moe_confirmatory_v2 import SAFETY
from bigan.v8.polymarket.regime_adaptive_lineage import REPO_ROOT

LINEAGE_ID = "BTC-15M-MoE-confirmatory-v2"
DEFAULT_CONFIG_DIR = (
    REPO_ROOT / "examples/v8/polymarket_configs/BTC-15M-MoE-confirmatory-v2"
)
DEFAULT_CONTRACT = (
    DEFAULT_CONFIG_DIR
    / "terminal_diagnostic_002/terminal_diagnostic_contract.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "terminal_diagnostic_002"


def generate_terminal_diagnostic(
    *,
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Generate scored rows and the report once from hash-bound source artifacts."""

    root = Path(repository_root).resolve()
    contract_file = Path(contract_path).resolve()
    output = Path(output_dir).resolve()
    if not contract_file.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("terminal diagnostic paths must remain repository-local")
    contract = _load_json(contract_file)
    _validate_contract(contract, repository_root=root)
    _assert_new_outputs_absent(output)

    scored_rows = recompute_terminal_scored_rows(
        contract=contract,
        repository_root=root,
    )
    scored_path = output / "terminal_diagnostic_scored_rows.jsonl"
    _write_new_jsonl(scored_path, scored_rows)
    scored_descriptor = _descriptor(scored_path, root)

    sources = _load_report_sources(contract, repository_root=root)
    report = build_terminal_diagnostic_report(
        contract=contract,
        scored_rows=scored_rows,
        scored_rows_descriptor=scored_descriptor,
        sources=sources,
    )
    report_path = output / "confirmatory_failure_diagnostic_report.json"
    report_artifact = _write_new_frozen_json(report_path, report)
    markdown_path = output / "confirmatory_failure_diagnostic_report.md"
    markdown_artifact = _write_new_frozen_text(
        markdown_path,
        render_terminal_diagnostic_markdown(report),
    )
    manifest = {
        "schema_version": "bigan-btc-15m-moe-terminal-diagnostic-manifest-v2",
        "lineage_id": LINEAGE_ID,
        "created_at": contract["created_at"],
        "contract": _descriptor(contract_file, root),
        "scored_rows": scored_descriptor,
        "report": _descriptor(Path(report_artifact["path"]), root),
        "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        "deterministic_generation": True,
        "terminal_result_preserved": report["terminal_result"]["terminal_failed"],
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    manifest_artifact = _write_new_frozen_json(
        output / "terminal_diagnostic_manifest.json",
        manifest,
    )
    return {
        "scored_rows": scored_descriptor,
        "report": _descriptor(Path(report_artifact["path"]), root),
        "report_markdown": _descriptor(Path(markdown_artifact["path"]), root),
        "manifest": _descriptor(Path(manifest_artifact["path"]), root),
        "terminal_failed": True,
    }


def verify_frozen_terminal_diagnostic(
    *,
    contract_path: Path | str = DEFAULT_CONTRACT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    repository_root: Path | str = REPO_ROOT,
    recompute_scored_rows_from_raw: bool = False,
) -> dict[str, Any]:
    """Independently rebuild the report and optionally all scored rows."""

    root = Path(repository_root).resolve()
    contract_file = Path(contract_path).resolve()
    output = Path(output_dir).resolve()
    contract = _load_json(contract_file)
    _validate_contract(contract, repository_root=root)
    manifest = _verified_json(output / "terminal_diagnostic_manifest.json")
    for field in ("contract", "scored_rows", "report", "report_markdown"):
        _verify_descriptor(manifest[field], repository_root=root)
    scored_path = _verify_descriptor(manifest["scored_rows"], repository_root=root)
    frozen_scored = _load_jsonl(scored_path)
    if recompute_scored_rows_from_raw:
        recomputed = recompute_terminal_scored_rows(
            contract=contract,
            repository_root=root,
        )
        if recomputed != frozen_scored:
            raise ValueError("terminal scored rows do not reproduce from raw evidence")
    sources = _load_report_sources(contract, repository_root=root)
    rebuilt = build_terminal_diagnostic_report(
        contract=contract,
        scored_rows=frozen_scored,
        scored_rows_descriptor=dict(manifest["scored_rows"]),
        sources=sources,
    )
    report_path = _verify_descriptor(manifest["report"], repository_root=root)
    _assert_semantically_equal(
        rebuilt,
        _load_json(report_path),
        path="terminal_report",
    )
    markdown_path = _verify_descriptor(
        manifest["report_markdown"], repository_root=root
    )
    if render_terminal_diagnostic_markdown(rebuilt) != markdown_path.read_text(
        encoding="utf-8"
    ):
        raise ValueError("terminal markdown does not reproduce from report JSON")
    return {
        "verification_passed": True,
        "scored_rows_recomputed_from_raw": recompute_scored_rows_from_raw,
        "scored_row_count": len(frozen_scored),
        "report_sha256": sha256_file(report_path),
        "markdown_sha256": sha256_file(markdown_path),
        "terminal_failed": rebuilt["terminal_result"]["terminal_failed"],
    }


def recompute_terminal_scored_rows(
    *,
    contract: Mapping[str, Any],
    repository_root: Path,
) -> list[dict[str, Any]]:
    """Recompute every market/decision/side score from recovered raw evidence."""

    config = repository_root / "examples/v8/polymarket_configs" / LINEAGE_ID
    freeze = config / "confirmatory_collection_freeze_001"
    artifacts = _evaluation_artifacts(freeze=freeze, config=config)
    contexts = _load_exact_contexts(
        repository_root=repository_root,
        artifacts=artifacts,
    )
    bundle = _load_runtime_bundle(repository_root)
    evaluation_rows = {
        row["market_id"]: row
        for row in _load_jsonl(
            _source_path(contract, "confirmatory_market_evaluation_rows", repository_root)
        )
    }
    rows: list[dict[str, Any]] = []
    for market_position, context in enumerate(contexts, start=1):
        market_id = str(context["market_id"])
        evaluation = evaluation_rows[market_id]
        outcome = str(evaluation["official_settlement"]["resolved_outcome"])
        candidate = dict(context["candidate"])
        baseline = dict(context["baseline"])
        for decision_position, feature_row in enumerate(
            sorted(context["feature_rows"], key=lambda row: int(row["decision_ts"])),
            start=1,
        ):
            router = _router_observation(feature_row, bundle["router"])
            route = deterministic_moe_route(router["router_inputs"])
            support = int(bundle["route_support"][route])
            actual_model = frozen_expert_or_fallback(
                route=route,
                expert_training_market_count=support,
            )
            candidate_model = (
                bundle["fallback"]
                if actual_model == "global_baseline_fallback"
                else bundle["experts"][route]
            )
            matrix = _feature_matrix(feature_row)
            candidate_probabilities = _pair_normalized_predictions(
                candidate_model,
                matrix,
                feature_names=bundle["feature_names"],
            )
            baseline_probabilities = _pair_normalized_predictions(
                bundle["fallback"],
                matrix,
                feature_names=bundle["feature_names"],
            )
            execution_costs = _execution_costs(feature_row)
            raw_features = dict(feature_row["features"])
            for side_index, side in enumerate(("UP", "DOWN")):
                prefix = side.lower()
                payout = 1.0 if outcome == side else 0.0
                ask = float(raw_features[f"{prefix}_ask"])
                bid = float(raw_features[f"{prefix}_bid"])
                depth = float(raw_features[f"{prefix}_liquidity_depth"])
                fees = 0.0002
                slippage = max(0.0001, (ask - bid) / 2.0)
                liquidity_impact = 0.00005 if depth > 0.0 else 0.001
                action_value = payout - ask - fees - slippage - liquidity_impact
                candidate_selected = _decision_selects(
                    candidate,
                    decision_ts=int(feature_row["decision_ts"]),
                    side=side,
                )
                baseline_selected = _decision_selects(
                    baseline,
                    decision_ts=int(feature_row["decision_ts"]),
                    side=side,
                )
                rows.append(
                    {
                        "schema_version": (
                            "bigan-btc-15m-moe-terminal-diagnostic-scored-row-v2"
                        ),
                        "lineage_id": LINEAGE_ID,
                        "market_id": market_id,
                        "market_position": market_position,
                        "market_start_ts": int(context["market"]["market_start_ts"]),
                        "decision_position": decision_position,
                        "decision_ts": int(feature_row["decision_ts"]),
                        "side": side,
                        "target": payout,
                        "resolved_outcome": outcome,
                        "requested_route": route,
                        "actual_model_used": actual_model,
                        "expert_training_market_count": support,
                        "expert_available": bool(bundle["route_available"][route]),
                        "fallback_used": actual_model == "global_baseline_fallback",
                        "candidate_pair_normalized_probability": float(
                            candidate_probabilities[side_index]
                        ),
                        "baseline_pair_normalized_probability": float(
                            baseline_probabilities[side_index]
                        ),
                        "candidate_predicted_net_score": float(
                            candidate_probabilities[side_index]
                            - execution_costs[side_index]
                        ),
                        "baseline_predicted_net_score": float(
                            baseline_probabilities[side_index]
                            - execution_costs[side_index]
                        ),
                        "candidate_action_selected": candidate_selected,
                        "baseline_action_selected": baseline_selected,
                        "realized_unit_net_pnl_if_action": action_value,
                        "cost_decomposition": {
                            "entry_ask": ask,
                            "entry_bid": bid,
                            "fees": fees,
                            "slippage": slippage,
                            "liquidity_impact": liquidity_impact,
                            "total_cost_excluding_entry_ask": (
                                fees + slippage + liquidity_impact
                            ),
                        },
                        "feature_row_sha256": canonical_json_sha256(feature_row),
                        "probability_clipping_applied_to_score": False,
                        "development_only_forever": True,
                        "promotion_evidence_eligible": False,
                        "safety": dict(SAFETY),
                    }
                )
    _validate_scored_rows(rows, contract=contract, evaluation_rows=evaluation_rows)
    return rows


def build_terminal_diagnostic_report(
    *,
    contract: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
    scored_rows_descriptor: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete report solely from frozen scored and summary inputs."""

    _validate_scored_rows(
        scored_rows,
        contract=contract,
        evaluation_rows={
            row["market_id"]: row for row in sources["evaluation_rows"]
        },
    )
    evaluation_report = sources["evaluation_report"]
    evaluation_rows = list(sources["evaluation_rows"])
    development_rows = list(sources["development_rows"])
    development_metric = _development_candidate_metric(
        sources["development_metric"]
    )
    development_distribution = list(
        sources["development_distribution"]["decision_time_rows"]
    )
    candidate_probability = _probability_metrics(
        scored_rows,
        probability_field="candidate_pair_normalized_probability",
        epsilon=float(
            contract["diagnostic_semantics"][
                "probability_clipping_epsilon_for_log_loss"
            ]
        ),
    )
    baseline_probability = _probability_metrics(
        scored_rows,
        probability_field="baseline_pair_normalized_probability",
        epsilon=float(
            contract["diagnostic_semantics"][
                "probability_clipping_epsilon_for_log_loss"
            ]
        ),
    )
    selected_candidate = _selected_action_metrics(scored_rows, policy="candidate")
    selected_baseline = _selected_action_metrics(scored_rows, policy="baseline")
    confirm_candidate = np.asarray(
        [row["candidate_unit_net_pnl"] for row in evaluation_rows], dtype=float
    )
    confirm_baseline = np.asarray(
        [row["baseline_unit_net_pnl"] for row in evaluation_rows], dtype=float
    )
    confirm_delta = confirm_candidate - confirm_baseline
    dev_candidate = np.asarray(
        [row["moe_unit_net_pnl"] for row in development_rows], dtype=float
    )
    dev_delta = np.asarray(
        [row["paired_delta_unit_net_pnl"] for row in development_rows], dtype=float
    )
    z_confidence = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.8)
    plugin_counts = {
        "target_crossing_probability_0_5": {
            "absolute_candidate": _plugin_sample_count(
                mean=float(np.mean(confirm_candidate)),
                standard_deviation=float(np.std(confirm_candidate, ddof=1)),
                z_confidence=z_confidence,
                z_target=0.0,
            ),
            "paired_delta": _plugin_sample_count(
                mean=float(np.mean(confirm_delta)),
                standard_deviation=float(np.std(confirm_delta, ddof=1)),
                z_confidence=z_confidence,
                z_target=0.0,
            ),
        },
        "target_crossing_probability_0_8": {
            "absolute_candidate": _plugin_sample_count(
                mean=float(np.mean(confirm_candidate)),
                standard_deviation=float(np.std(confirm_candidate, ddof=1)),
                z_confidence=z_confidence,
                z_target=z_power,
            ),
            "paired_delta": _plugin_sample_count(
                mean=float(np.mean(confirm_delta)),
                standard_deviation=float(np.std(confirm_delta, ddof=1)),
                z_confidence=z_confidence,
                z_target=z_power,
            ),
        },
    }
    if plugin_counts != {
        key.removeprefix("plugin_market_count_"): dict(value)
        for key, value in contract["power_diagnostic_semantics"].items()
        if key.startswith("plugin_market_count_")
    }:
        raise ValueError("power diagnostic counts differ from frozen semantics")

    route_diagnosis = {
        route: {
            "development": _panel_metrics(
                [row for row in development_rows if row["requested_route"] == route],
                candidate_field="moe_unit_net_pnl",
                delta_field="paired_delta_unit_net_pnl",
            ),
            "confirmatory": _panel_metrics(
                [row for row in evaluation_rows if row["requested_route"] == route],
                candidate_field="candidate_unit_net_pnl",
                delta_field="paired_delta_unit_net_pnl",
            ),
        }
        for route in ("high_vol", "bullish", "bearish", "low_vol")
    }
    midpoint = len(evaluation_rows) // 2
    quartiles = np.array_split(np.arange(len(evaluation_rows)), 4)
    report = {
        "schema_version": "bigan-btc-15m-moe-terminal-failure-diagnostic-v2",
        "lineage_id": LINEAGE_ID,
        "created_at": contract["created_at"],
        "role": "deterministic_post_terminal_failure_diagnostic",
        "contract_sha256": canonical_json_sha256(contract),
        "scored_rows": dict(scored_rows_descriptor),
        "source_artifacts": dict(contract["source_artifacts"]),
        "population_denominators": dict(contract["population_denominators"]),
        "terminal_result": {
            "market_count": len(evaluation_rows),
            "candidate_total_unit_net_pnl": float(np.sum(confirm_candidate)),
            "baseline_total_unit_net_pnl": float(np.sum(confirm_baseline)),
            "paired_delta_total_unit_net_pnl": float(np.sum(confirm_delta)),
            "candidate_bootstrap_interval": evaluation_report["panels"]["overall"][
                "candidate_bootstrap_interval"
            ],
            "paired_delta_bootstrap_interval": evaluation_report["panels"][
                "overall"
            ]["paired_delta_bootstrap_interval"],
            "confirmatory_gate_passed": False,
            "terminal_failed": True,
            "rerun_allowed": False,
        },
        "effect_reconciliation": {
            "candidate": _effect_comparison(dev_candidate, confirm_candidate),
            "paired_delta": _effect_comparison(dev_delta, confirm_delta),
            "development_effect_population_count": len(development_rows),
            "confirmatory_population_count": len(evaluation_rows),
        },
        "probability_metrics": {
            "candidate": candidate_probability,
            "matched_global_baseline": baseline_probability,
            "probability_clipping_epsilon_for_log_loss": contract[
                "diagnostic_semantics"
            ]["probability_clipping_epsilon_for_log_loss"],
        },
        "selected_action_metrics": {
            "candidate": selected_candidate,
            "matched_global_baseline": selected_baseline,
            "correlation_method": contract["diagnostic_semantics"]["correlation"],
        },
        "cost_signal": {
            "development_cost_signal_ratio": development_metric["trading_metrics"][
                "cost_signal_ratio"
            ],
            "confirmatory": _confirmatory_cost_signal(evaluation_rows),
            "primary_failure_is_cost_increase": False,
            "interpretation": "realized_gross_edge_collapsed_relative_to_stable_frozen_costs",
        },
        "route_diagnosis": route_diagnosis,
        "fallback_attribution_caveat": _fallback_attribution(evaluation_rows, scored_rows),
        "temporal_diagnosis": {
            "candidate_chronological_halves": {
                "first": float(np.sum(confirm_candidate[:midpoint])),
                "second": float(np.sum(confirm_candidate[midpoint:])),
            },
            "paired_delta_chronological_halves": {
                "first": float(np.sum(confirm_delta[:midpoint])),
                "second": float(np.sum(confirm_delta[midpoint:])),
            },
            "candidate_quartile_totals": [
                float(np.sum(confirm_candidate[index])) for index in quartiles
            ],
            "paired_delta_quartile_totals": [
                float(np.sum(confirm_delta[index])) for index in quartiles
            ],
            "stable_absolute_and_relative_edge_in_same_half": False,
        },
        "side_diagnosis": _side_diagnosis(evaluation_rows),
        "distribution_shift": _distribution_shift(
            development_distribution,
            evaluation_rows,
        ),
        "power_diagnostic": {
            **plugin_counts,
            "target_crossing_probability_0_5_is_not_80pct_power": True,
            "diagnostic_only_not_collection_plan": True,
        },
        "root_cause": {
            "collection_or_settlement_defect_identified": False,
            "frozen_derived_artifact_defect_identified": False,
            "raw_provenance_status": contract.get(
                "raw_provenance_status",
                "not_independently_verifiable_from_exact_git_tree_until_archive_restored",
            ),
            "development_effect_optimism": True,
            "market_relative_edge_ranking_failure": True,
            "route_specific_non_generalization": True,
            "temporal_instability": True,
            "distribution_shift_contributed": True,
            "conclusion": (
                "raw outcome probability retained direction signal but did not provide "
                "stable incremental edge over executable market price after frozen costs"
            ),
        },
        "new_lineage_decision": {
            "recommended_lineage_id": "BTC-15M-cost-aware-market-residual-v1",
            "development_justification": "conditionally_justified",
            "required_hypothesis_change": (
                "pooled side-symmetric direct regression of market-relative after-cost "
                "action value"
            ),
            "creation_authorized_by_this_report": False,
            "training_authorized_by_this_report": False,
            "confirmatory_collection_authorized": False,
        },
        "future_evidence_boundary": dict(contract["future_evidence_boundary"]),
        "promotion_evidence_eligible": False,
        "safety": dict(SAFETY),
    }
    _validate_report(report, contract=contract)
    return report


def render_terminal_diagnostic_markdown(report: Mapping[str, Any]) -> str:
    terminal = report["terminal_result"]
    candidate = report["probability_metrics"]["candidate"]
    baseline = report["probability_metrics"]["matched_global_baseline"]
    selected = report["selected_action_metrics"]["candidate"]
    effect = report["effect_reconciliation"]
    power = report["power_diagnostic"]
    lines = [
        "# BTC-15M-MoE-confirmatory-v2 terminal failure diagnostic v2",
        "",
        "## Terminal decision",
        "",
        "The lineage remains `terminal_failed`. No rerun, extension, threshold rescue, "
        "paper/live unlock, or promotion is allowed.",
        "",
        "A materially new development lineage is conditionally justified: "
        "`BTC-15M-cost-aware-market-residual-v1`. This report authorizes neither "
        "training nor collection.",
        "",
        "## Population denominators",
        "",
        f"- Distribution diagnostics: `{report['population_denominators']['distribution_population_count']}` markets",
        f"- Development effect/power diagnostics: `{report['population_denominators']['effect_population_count']}` OOF markets",
        f"- Confirmatory result: `{report['population_denominators']['confirmatory_population_count']}` markets",
        "",
        "## Confirmatory result",
        "",
        f"- Candidate total unit PnL: `{terminal['candidate_total_unit_net_pnl']:.8f}`",
        f"- Baseline total unit PnL: `{terminal['baseline_total_unit_net_pnl']:.8f}`",
        f"- Paired delta: `{terminal['paired_delta_total_unit_net_pnl']:.8f}`",
        f"- Candidate LCB: `{terminal['candidate_bootstrap_interval']['lower']:.8f}`",
        f"- Paired-delta LCB: `{terminal['paired_delta_bootstrap_interval']['lower']:.8f}`",
        "",
        "## Effect and economic-edge diagnosis",
        "",
        f"- Candidate development effect retained: `{effect['candidate']['effect_retained_fraction']:.4%}`",
        f"- Paired-delta development effect retained: `{effect['paired_delta']['effect_retained_fraction']:.4%}`",
        f"- Candidate AUC/Brier/log loss: `{candidate['auc']:.6f}` / `{candidate['brier_score']:.6f}` / `{candidate['log_loss']:.6f}`",
        f"- Baseline AUC/Brier/log loss: `{baseline['auc']:.6f}` / `{baseline['brier_score']:.6f}` / `{baseline['log_loss']:.6f}`",
        f"- Candidate mean predicted net score: `{selected['mean_predicted_net_score']:.8f}`",
        f"- Candidate realized mean unit PnL: `{selected['mean_realized_unit_net_pnl']:.8f}`",
        f"- Score-to-PnL Pearson correlation: `{selected['score_to_realized_pnl_correlation']:.8f}`",
        "",
        "## Power wording",
        "",
        f"The plug-in counts `{power['target_crossing_probability_0_5']['absolute_candidate']}` / "
        f"`{power['target_crossing_probability_0_5']['paired_delta']}` target an approximate "
        "50% chance of LCB crossing, not 80% power. The corresponding diagnostic 80% "
        f"counts are `{power['target_crossing_probability_0_8']['absolute_candidate']}` / "
        f"`{power['target_crossing_probability_0_8']['paired_delta']}`. None is a collection plan.",
        "",
        "## Integrity boundary",
        "",
        "No defect was identified in frozen derived artifacts. Raw provenance is only "
        "independently replayable when the separately hash-bound recovered archive is "
        "available and verified.",
        "",
        "The opened 800 markets, 113-market distribution corpus, and 73-market OOF "
        "effect corpus are development-only forever and cannot become future promotion evidence.",
        "",
        "All safety permissions remain false.",
        "",
    ]
    return "\n".join(lines)


def _probability_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_field: str,
    epsilon: float,
) -> dict[str, Any]:
    probability = np.asarray([row[probability_field] for row in rows], dtype=float)
    target = np.asarray([row["target"] for row in rows], dtype=float)
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    positive = probability[target == 1.0]
    negative = probability[target == 0.0]
    pairwise = (
        np.sum(positive[:, None] > negative)
        + 0.5 * np.sum(positive[:, None] == negative)
    )
    return {
        "side_decision_row_count": len(rows),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            np.mean(
                -(
                    target * np.log(clipped)
                    + (1.0 - target) * np.log(1.0 - clipped)
                )
            )
        ),
        "auc": float(pairwise / (len(positive) * len(negative))),
        "probability_clipping_rule": f"clip_to_[{epsilon},1-{epsilon}]_for_log_loss_only",
    }


def _selected_action_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row[f"{policy}_action_selected"]]
    scores = np.asarray(
        [row[f"{policy}_predicted_net_score"] for row in selected], dtype=float
    )
    pnl = np.asarray(
        [row["realized_unit_net_pnl_if_action"] for row in selected], dtype=float
    )
    return {
        "accepted_market_count": len(selected),
        "mean_predicted_net_score": float(np.mean(scores)),
        "mean_realized_unit_net_pnl": float(np.mean(pnl)),
        "prediction_optimism_gap": float(np.mean(scores - pnl)),
        "score_to_realized_pnl_correlation": float(np.corrcoef(scores, pnl)[0, 1]),
        "win_rate": float(np.mean(pnl > 0.0)),
    }


def _effect_comparison(
    development: np.ndarray,
    confirmatory: np.ndarray,
) -> dict[str, Any]:
    development_mean = float(np.mean(development))
    confirmatory_mean = float(np.mean(confirmatory))
    return {
        "development_mean_unit_net_pnl": development_mean,
        "confirmatory_mean_unit_net_pnl": confirmatory_mean,
        "confirmatory_standard_deviation": float(np.std(confirmatory, ddof=1)),
        "effect_retained_fraction": confirmatory_mean / development_mean,
        "effect_shrinkage_fraction": 1.0 - confirmatory_mean / development_mean,
    }


def _plugin_sample_count(
    *,
    mean: float,
    standard_deviation: float,
    z_confidence: float,
    z_target: float,
) -> int:
    if mean <= 0.0 or standard_deviation <= 0.0:
        raise ValueError("positive mean and standard deviation required for power diagnostic")
    return math.ceil(((z_confidence + z_target) * standard_deviation / mean) ** 2)


def _panel_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_field: str,
    delta_field: str,
) -> dict[str, Any]:
    candidate = np.asarray([row[candidate_field] for row in rows], dtype=float)
    delta = np.asarray([row[delta_field] for row in rows], dtype=float)
    return {
        "market_count": len(rows),
        "candidate_total_unit_net_pnl": float(np.sum(candidate)),
        "candidate_mean_unit_net_pnl": float(np.mean(candidate)),
        "paired_delta_total_unit_net_pnl": float(np.sum(delta)),
        "paired_delta_mean_unit_net_pnl": float(np.mean(delta)),
    }


def _confirmatory_cost_signal(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["candidate_accepted"]]
    gross = sum(
        float(row["cost_decomposition"]["candidate"]["gross_price_edge"])
        for row in accepted
    )
    cost = sum(
        float(row["cost_decomposition"]["candidate"]["total_cost"])
        for row in accepted
    )
    return {
        "accepted_market_count": len(accepted),
        "gross_price_edge_total": gross,
        "total_cost": cost,
        "cost_signal_ratio": cost / gross,
    }


def _fallback_attribution(
    evaluation_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fallback_markets = {
        row["market_id"] for row in evaluation_rows if row["fallback_used"]
    }
    by_market: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        if row["market_id"] in fallback_markets:
            by_market[row["market_id"]].append(row)
    identical = 0
    sequential = 0
    for rows in by_market.values():
        candidate = [row for row in rows if row["candidate_action_selected"]]
        baseline = [row for row in rows if row["baseline_action_selected"]]
        if len(candidate) == len(baseline) == 1 and (
            candidate[0]["decision_ts"], candidate[0]["side"]
        ) == (baseline[0]["decision_ts"], baseline[0]["side"]):
            identical += 1
        elif len(candidate) == len(baseline) == 1 and candidate[0]["fallback_used"]:
            sequential += 1
        elif not candidate and not baseline:
            identical += 1
    return {
        "fallback_attributed_market_count": len(fallback_markets),
        "identical_candidate_and_baseline_action_count": identical,
        "sequential_expert_rejection_then_fallback_acceptance_count": sequential,
        "fallback_panel_paired_delta": sum(
            float(row["paired_delta_unit_net_pnl"])
            for row in evaluation_rows
            if row["fallback_used"]
        ),
        "direct_fallback_superiority_claim_allowed": False,
    }


def _side_diagnosis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for side in ("UP", "DOWN"):
        selected = [row for row in rows if row["candidate_selected_side"] == side]
        output[side] = {
            "accepted_market_count": len(selected),
            "total_unit_net_pnl": sum(
                float(row["candidate_unit_net_pnl"]) for row in selected
            ),
            "mean_unit_net_pnl": float(
                np.mean([row["candidate_unit_net_pnl"] for row in selected])
            ),
        }
    outcomes = Counter(
        row["official_settlement"]["resolved_outcome"] for row in rows
    )
    output["outcome_distribution"] = {
        side: {"count": outcomes[side], "share": outcomes[side] / len(rows)}
        for side in ("UP", "DOWN")
    }
    return output


def _distribution_shift(
    development: Sequence[Mapping[str, Any]],
    confirmatory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def ratio(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float:
        return sum(bool(predicate(row)) for row in rows) / len(rows)

    return {
        "distribution_population_count": len(development),
        "confirmatory_population_count": len(confirmatory),
        "high_vol_route_share": {
            "development": ratio(
                development, lambda row: row["requested_route"] == "high_vol"
            ),
            "confirmatory": ratio(
                confirmatory, lambda row: row["requested_route"] == "high_vol"
            ),
        },
        "low_vol_route_share": {
            "development": ratio(
                development, lambda row: row["requested_route"] == "low_vol"
            ),
            "confirmatory": ratio(
                confirmatory, lambda row: row["requested_route"] == "low_vol"
            ),
        },
        "fallback_share": {
            "development": ratio(development, lambda row: row["fallback_used"]),
            "confirmatory": ratio(confirmatory, lambda row: row["fallback_used"]),
        },
        "feature_complete_share": {
            "development": ratio(
                development, lambda row: row["missing_feature_count"] == 0
            ),
            "confirmatory": ratio(
                confirmatory,
                lambda row: row["feature_missingness"]["feature_complete"],
            ),
        },
    }


def _validate_scored_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    evaluation_rows: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = int(
        contract["diagnostic_semantics"]["probability_metrics_side_row_count"]
    )
    if len(rows) != expected:
        raise ValueError("terminal diagnostic scored-row count mismatch")
    identities = [
        (
            int(row["market_position"]),
            int(row["decision_position"]),
            str(row["side"]),
        )
        for row in rows
    ]
    expected_identities = [
        (market_position, decision_position, side)
        for market_position in range(1, 801)
        for decision_position in (1, 2)
        for side in ("UP", "DOWN")
    ]
    if identities != expected_identities:
        raise ValueError("terminal diagnostic scored rows were dropped or reordered")
    if any(
        row["promotion_evidence_eligible"] is not False
        or row["development_only_forever"] is not True
        or any(row["safety"].values())
        for row in rows
    ):
        raise ValueError("terminal diagnostic scored rows crossed safety boundary")
    candidate_selected = [row for row in rows if row["candidate_action_selected"]]
    baseline_selected = [row for row in rows if row["baseline_action_selected"]]
    if len(candidate_selected) != 795 or len(baseline_selected) != 792:
        raise ValueError("terminal selected-action counts changed")
    for policy, selected in (
        ("candidate", candidate_selected),
        ("baseline", baseline_selected),
    ):
        for row in selected:
            expected_pnl = evaluation_rows[row["market_id"]][
                f"{policy}_unit_net_pnl"
            ]
            if not math.isclose(
                float(row["realized_unit_net_pnl_if_action"]),
                float(expected_pnl),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("scored action PnL differs from terminal evaluation")


def _validate_contract(contract: Mapping[str, Any], *, repository_root: Path) -> None:
    if contract.get("lineage_id") != LINEAGE_ID:
        raise ValueError("terminal diagnostic contract lineage mismatch")
    if contract.get("terminal_invariants") != {
        "additional_v2_collection_allowed": False,
        "confirmatory_gate_passed": False,
        "rerun_allowed": False,
        "terminal_failed": True,
        "v2_permission_change_allowed": False,
    }:
        raise ValueError("terminal invariants changed")
    if contract.get("safety") != SAFETY or any(contract["safety"].values()):
        raise ValueError("terminal diagnostic contract safety changed")
    for descriptor in contract["source_artifacts"].values():
        _verify_descriptor(descriptor, repository_root=repository_root)


def _validate_report(report: Mapping[str, Any], *, contract: Mapping[str, Any]) -> None:
    terminal = report["terminal_result"]
    if terminal["terminal_failed"] is not True:
        raise ValueError("terminal report did not preserve failure")
    if terminal["confirmatory_gate_passed"] is not False:
        raise ValueError("terminal report changed confirmatory result")
    if terminal["rerun_allowed"] is not False:
        raise ValueError("terminal report allowed rerun")
    if report["population_denominators"] != contract["population_denominators"]:
        raise ValueError("terminal report population denominators changed")
    if report["promotion_evidence_eligible"] is not False:
        raise ValueError("terminal report became promotion evidence")
    if report["safety"] != SAFETY or any(report["safety"].values()):
        raise ValueError("terminal report safety changed")


def _load_report_sources(
    contract: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "evaluation_report": _load_json(
            _source_path(contract, "confirmatory_evaluation_report", repository_root)
        ),
        "evaluation_rows": _load_jsonl(
            _source_path(
                contract, "confirmatory_market_evaluation_rows", repository_root
            )
        ),
        "development_rows": _load_jsonl(
            _source_path(contract, "development_paired_planning_rows", repository_root)
        ),
        "development_metric": _load_json(
            _source_path(contract, "development_metric_reconciliation", repository_root)
        ),
        "development_distribution": _load_json(
            _source_path(contract, "development_distribution_reference", repository_root)
        ),
    }


def _development_candidate_metric(payload: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in payload["candidate_reconciliation"]
        if row["candidate_id"] == "mixture_of_experts"
    ]
    if len(matches) != 1 or matches[0]["comparison"]["passed"] is not True:
        raise ValueError("development candidate reconciliation unavailable")
    return dict(matches[0]["recomputed"])


def _decision_selects(
    decision: Mapping[str, Any],
    *,
    decision_ts: int,
    side: str,
) -> bool:
    return bool(
        decision["accepted"]
        and int(decision["decision_ts"]) == decision_ts
        and decision["selected_side"] == side
    )


def _source_path(
    contract: Mapping[str, Any],
    name: str,
    repository_root: Path,
) -> Path:
    return _verify_descriptor(
        contract["source_artifacts"][name],
        repository_root=repository_root,
    )


def _verify_descriptor(
    descriptor: Mapping[str, Any],
    *,
    repository_root: Path,
) -> Path:
    path = (repository_root / str(descriptor["path"])).resolve()
    if not path.is_relative_to(repository_root):
        raise ValueError("terminal diagnostic descriptor escaped repository")
    if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
        raise ValueError(f"terminal diagnostic source SHA mismatch: {path}")
    return path


def _verified_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError(f"missing frozen JSON artifact: {path}")
    if sha256_file(path) != sidecar.read_text(encoding="utf-8").strip():
        raise ValueError(f"frozen JSON sidecar mismatch: {path}")
    return _load_json(path)


def _descriptor(path: Path, repository_root: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("terminal diagnostic output escaped repository")
    return {
        "path": resolved.relative_to(repository_root).as_posix(),
        "sha256": sha256_file(resolved),
    }


def _assert_new_outputs_absent(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "terminal_diagnostic_scored_rows.jsonl",
        "terminal_diagnostic_scored_rows.sha256",
        "confirmatory_failure_diagnostic_report.json",
        "confirmatory_failure_diagnostic_report.sha256",
        "confirmatory_failure_diagnostic_report.md",
        "confirmatory_failure_diagnostic_report.md.sha256",
        "terminal_diagnostic_manifest.json",
        "terminal_diagnostic_manifest.sha256",
    ):
        if (output / name).exists():
            raise FileExistsError(f"terminal diagnostic output already exists: {name}")


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


def _assert_semantically_equal(
    actual: Any,
    expected: Any,
    *,
    path: str,
    relative_tolerance: float = 1e-12,
    absolute_tolerance: float = 1e-12,
) -> None:
    """Compare deterministic reports while absorbing platform float roundoff only."""

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise ValueError(f"terminal report field set changed at {path}")
        for key in actual:
            _assert_semantically_equal(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if (
        isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
        and isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
    ):
        if len(actual) != len(expected):
            raise ValueError(f"terminal report sequence length changed at {path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_semantically_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if (
        isinstance(actual, float)
        and isinstance(expected, float)
        and math.isfinite(actual)
        and math.isfinite(expected)
    ):
        if not math.isclose(
            actual,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"terminal report numeric value changed at {path}: "
                f"{actual!r} != {expected!r}"
            )
        return
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(
            f"terminal report value changed at {path}: {actual!r} != {expected!r}"
        )


__all__ = [
    "build_terminal_diagnostic_report",
    "generate_terminal_diagnostic",
    "recompute_terminal_scored_rows",
    "render_terminal_diagnostic_markdown",
    "verify_frozen_terminal_diagnostic",
]
