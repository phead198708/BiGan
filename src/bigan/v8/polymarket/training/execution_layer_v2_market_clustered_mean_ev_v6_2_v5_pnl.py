"""Split-aware retrospective PnL attribution for frozen v6.2 on v5 data."""

from __future__ import annotations

import hashlib
import math
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
    apply_conformal_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2 import (
    CANDIDATE_NAME,
    apply_market_clustered_mean_ev_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_acceptance_viability import (
    _outcome_blind_acceptance_replay,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-v5-pnl"
ROLE_GROUPS = {
    "model_fit_135": {
        "manifest_key": "development_train_action_rows",
        "source_roles": ("development_train", "development_calibration"),
    },
    "mean_risk_calibration_60": {
        "manifest_key": "development_calibration_action_rows",
        "source_roles": ("confirmatory_validation",),
    },
}


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62V5PnLConfig:
    """Pinned source artifacts for one retrospective report."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    v5_freeze_manifest_path: Path | str
    expected_v5_freeze_manifest_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    implementation_commit: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for value, name in (
            (self.expected_profile_sha256, "profile sha256"),
            (self.expected_candidate_manifest_sha256, "candidate manifest sha256"),
            (self.expected_v5_freeze_manifest_sha256, "v5 freeze manifest sha256"),
            (self.expected_feature_contract_sha256, "feature contract sha256"),
        ):
            _require_sha256(value, name=name)
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "profile_path",
            "candidate_manifest_path",
            "v5_freeze_manifest_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_market_clustered_mean_ev_v6_2_v5_pnl(
    config: MarketClusteredMeanEVV62V5PnLConfig,
) -> dict[str, Any]:
    """Replay frozen v6.2 and fixed baselines on settled v5 roles."""

    profile_path = config.profile_path.resolve()
    candidate_path = config.candidate_manifest_path.resolve()
    v5_path = config.v5_freeze_manifest_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
    _verify_pin(profile_path, config.expected_profile_sha256, "#213 profile")
    _verify_pin(
        candidate_path,
        config.expected_candidate_manifest_sha256,
        "v6.2 candidate manifest",
    )
    _verify_pin(v5_path, config.expected_v5_freeze_manifest_sha256, "v5 freeze manifest")
    _verify_pin(
        feature_contract_path,
        config.expected_feature_contract_sha256,
        "feature contract",
    )
    profile = _load_json(profile_path)
    candidate = _load_json(candidate_path)
    v5_manifest = _load_json(v5_path)
    _validate_profile(profile)
    _validate_candidate(candidate)
    if v5_manifest.get("research_candidate_frozen") is not True:
        raise ValueError("v5 source candidate is not frozen")
    model_descriptor = _verified_descriptor(candidate.get("source_model"), "source model")
    calibration_descriptor = _verified_descriptor(
        candidate.get("market_clustered_mean_risk_calibration"), "v6.2 calibration"
    )
    original_v5_calibration_descriptor = _verified_descriptor(
        v5_manifest.get("calibration_artifact"), "original v5 calibration"
    )
    original_v5_profile_descriptor = _verified_descriptor(
        v5_manifest.get("fit_profile"), "original v5 profile"
    )
    feature_contract = _load_json(feature_contract_path)
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])
    calibration = _load_json(Path(calibration_descriptor["path"]))
    original_v5_calibration = _load_json(Path(original_v5_calibration_descriptor["path"]))
    original_v5_profile = _load_json(Path(original_v5_profile_descriptor["path"]))
    booster = xgb.Booster()
    booster.load_model(model_descriptor["path"])

    all_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_descriptors = {}
    raw_predictions_by_role = {}
    source_role_market_ids: dict[str, set[str]] = defaultdict(set)
    for role, role_group in ROLE_GROUPS.items():
        manifest_key = str(role_group["manifest_key"])
        descriptor = _verified_descriptor(v5_manifest.get(manifest_key), f"v5 {role} rows")
        source_descriptors[role] = descriptor
        target_rows = _load_jsonl(Path(descriptor["path"]))
        _validate_target_rows(
            target_rows,
            expected_roles=tuple(role_group["source_roles"]),
        )
        for row in target_rows:
            source_role_market_ids[str(row["role"])].add(str(row["market_id"]))
        raw = _raw_target_stripped_predictions(
            booster,
            target_rows,
            feature_columns=feature_columns,
        )
        raw_predictions_by_role[role] = raw
        compatible = attach_frozen_execution_compatibility(raw)
        variants = {
            "market_clustered_mean_ev_v6_2": apply_market_clustered_mean_ev_scores(
                compatible,
                calibration_artifact=calibration,
            ),
            "raw_point_policy_diagnostic": _raw_point_scores(compatible),
            "original_v5_individual_outcome_conformal_policy": apply_conformal_scores(
                raw,
                calibration_artifact=original_v5_calibration,
                profile=original_v5_profile,
            ),
        }
        for policy_name, scored in variants.items():
            replay = _outcome_blind_acceptance_replay(
                _replay_score_adapter(scored),
                entry_threshold=0.0,
                runner_up_advantage_threshold=0.0,
            )
            accepted = [row for row in replay if row["execution_guard_order_allowed"]]
            all_evidence[policy_name].extend(
                _join_pnl_evidence(
                    accepted,
                    target_rows=target_rows,
                    role=role,
                    policy_name=policy_name,
                )
            )

    all_evidence["no_trade_zero"] = []
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    evidence_descriptors = {}
    for policy_name, rows in sorted(all_evidence.items()):
        path = run_dir / f"{policy_name}_accepted_pnl_rows.jsonl"
        _write_jsonl(path, rows)
        evidence_descriptors[policy_name] = _descriptor(path)
    raw_prediction_descriptors = {}
    for role, rows in raw_predictions_by_role.items():
        path = run_dir / f"{role}_target_stripped_raw_predictions.jsonl"
        _write_jsonl(path, rows)
        raw_prediction_descriptors[role] = _descriptor(path)

    policy_metrics = {
        policy_name: _policy_metrics(rows, profile=profile)
        for policy_name, rows in sorted(all_evidence.items())
    }
    candidate_metrics = policy_metrics["market_clustered_mean_ev_v6_2"]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "source_model_sha256": model_descriptor["sha256"],
        "mean_risk_calibration_sha256": calibration_descriptor["sha256"],
        "v5_source_market_count_by_role": {
            role: len(
                {
                    str(row["market_id"])
                    for row in _load_jsonl(Path(descriptor["path"]))
                }
            )
            for role, descriptor in source_descriptors.items()
        },
        "v5_source_market_count_by_source_role": {
            role: len(market_ids)
            for role, market_ids in sorted(source_role_market_ids.items())
        },
        "v5_source_total_unique_market_count": len(
            {
                str(row["market_id"])
                for descriptor in source_descriptors.values()
                for row in _load_jsonl(Path(descriptor["path"]))
            }
        ),
        "policy_metrics": policy_metrics,
        "candidate_after_cost_sized_net_pnl": candidate_metrics[
            "after_cost_sized_net_pnl"
        ],
        "candidate_after_cost_per_contract_net_pnl": candidate_metrics[
            "after_cost_per_contract_net_pnl"
        ],
        "candidate_cost_basis": candidate_metrics["cost_basis"],
        "candidate_roi": candidate_metrics["roi"],
        "candidate_accepted_unique_market_count": candidate_metrics[
            "accepted_unique_market_count"
        ],
        "candidate_pnl_positive": candidate_metrics["after_cost_sized_net_pnl"] > 0.0,
        "development_retrospective_only": True,
        "model_fit_role_is_in_sample": True,
        "confirmatory_role_used_for_v6_2_mean_risk_calibration": True,
        "independent_future_pnl_evidence": False,
        "result_used_to_tune_v6_2_or_future_gate": False,
        "future_holdout_required": True,
        "future_holdout_artifacts_opened": False,
        "promotion_evidence": False,
        "threshold_or_guard_tuning_performed": False,
        "model_or_source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "execution_layer_v2_v6_2_v5_retrospective_pnl_report.json"
    report_md_path = run_dir / "execution_layer_v2_v6_2_v5_retrospective_pnl_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "profile": _descriptor(profile_path),
        "candidate_manifest": _descriptor(candidate_path),
        "v5_freeze_manifest": _descriptor(v5_path),
        "feature_contract": _descriptor(feature_contract_path),
        "source_model": model_descriptor,
        "mean_risk_calibration": calibration_descriptor,
        "source_role_rows": source_descriptors,
        "target_stripped_raw_predictions": raw_prediction_descriptors,
        "accepted_pnl_rows": evidence_descriptors,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "development_retrospective_only": True,
        "future_holdout_artifacts_opened": False,
        "promotion_evidence": False,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "execution_layer_v2_v6_2_v5_retrospective_pnl_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _raw_point_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        action = str(row["action"])
        raw = 0.0 if action == "NO_TRADE" else float(row["raw_direct_predicted_net_return"])
        score = raw if row["guard_compatible_before_ranking"] else -1_000_000.0
        output.append(
            {
                **row,
                "conformal_net_return_lower_bound": raw,
                "action_selection_score": score,
                "action_advantage_lcb_net_return": score,
                "calibrated_action_expected_net_return": raw,
                "ranking_score_source": "raw_point_policy_diagnostic",
            }
        )
    return output


def _replay_score_adapter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "raw_pairwise_rank_score": float(row["raw_direct_predicted_net_return"]),
            "pairwise_group_normalized_rank_score": float(
                row["raw_direct_predicted_net_return"]
            ),
            "action_advantage_lcb_score_bucket": str(
                row.get("action_advantage_lcb_score_bucket") or "retrospective"
            ),
            "action_advantage_lcb_estimate_source": str(
                row.get("action_advantage_lcb_estimate_source")
                or row.get("ranking_score_source")
                or "retrospective"
            ),
        }
        for row in rows
    ]


def _join_pnl_evidence(
    accepted: list[dict[str, Any]],
    *,
    target_rows: list[dict[str, Any]],
    role: str,
    policy_name: str,
) -> list[dict[str, Any]]:
    targets = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in target_rows
    }
    if len(targets) != len(target_rows):
        raise ValueError("v5 target row identity is not unique")
    output = []
    for row in accepted:
        key = (
            str(row["market_id"]),
            int(row["decision_ts"]),
            str(row["executed_action"]),
        )
        target = targets.get(key)
        if target is None:
            raise ValueError("accepted action target row missing")
        order_size = float(row["proposed_order_size"])
        entry_ask = float(row["microstructure_snapshot"]["entry_ask"])
        per_contract = float(target["target_net_pnl_per_contract"])
        sized_pnl = per_contract * order_size
        evidence = {
            "policy_name": policy_name,
            "role": role,
            "market_id": key[0],
            "decision_ts": key[1],
            "action": key[2],
            "action_family": str(row["selected_action_family"]),
            "side": str(row["selected_side"]),
            "resolved_outcome": str(target["target_resolved_outcome"]),
            "execution_price": entry_ask,
            "order_size": order_size,
            "cost_basis": entry_ask * order_size,
            "target_net_pnl_per_contract": per_contract,
            "after_cost_sized_net_pnl": sized_pnl,
            "target_cost_components": dict(target["target_cost_components"]),
            "p_up": float(row["p_up"]),
            "p_down": float(row["p_down"]),
            "selected_score": float(row["decision_score"]),
            "time_to_close_seconds": float(
                row["microstructure_snapshot"]["time_to_close_seconds"]
            ),
            "spread_bps": float(row["microstructure_snapshot"]["spread_bps"]),
            "book_staleness_ms": float(
                row["microstructure_snapshot"]["book_staleness_ms"]
            ),
            "queue_fill_proxy": float(row["microstructure_snapshot"]["queue_fill_proxy"]),
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "promotion_evidence": False,
            **_blocked_safety_fields(),
        }
        evidence["evidence_row_sha256"] = canonical_json_sha256(evidence)
        output.append(evidence)
    return sorted(output, key=lambda value: (value["decision_ts"], value["market_id"]))


def _policy_metrics(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    roles = {
        role: _summary([row for row in rows if row["role"] == role], profile=profile)
        for role in ROLE_GROUPS
    }
    total = _summary(rows, profile=profile)
    return {
        **total,
        "by_role": roles,
        "pnl_by_side": _group_summaries(rows, "side"),
        "pnl_by_action": _group_summaries(rows, "action"),
        "pnl_by_action_family": _group_summaries(rows, "action_family"),
        "pnl_by_price_bucket": _group_summaries(rows, "price_bucket"),
        "pnl_by_time_to_close_bucket": _group_summaries(
            rows, "time_to_close_bucket"
        ),
    }


def _summary(rows: list[dict[str, Any]], *, profile: dict[str, Any]) -> dict[str, Any]:
    pnl = [float(row["after_cost_sized_net_pnl"]) for row in rows]
    per_contract = [float(row["target_net_pnl_per_contract"]) for row in rows]
    cost_basis = sum(float(row["cost_basis"]) for row in rows)
    total = sum(pnl)
    ordered = sorted(rows, key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in ordered:
        running += float(row["after_cost_sized_net_pnl"])
        peak = max(peak, running)
        max_drawdown = max(max_drawdown, peak - running)
    seed = int(profile["pnl"]["bootstrap_seed"])
    resamples = int(profile["pnl"]["bootstrap_resample_count"])
    confidence = float(profile["pnl"]["bootstrap_confidence_level"])
    ci = _bootstrap_mean_ci(pnl, seed=seed, resamples=resamples, confidence=confidence)
    largest_winner = max(pnl, default=0.0)
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "cost_basis": cost_basis,
        "after_cost_sized_net_pnl": total,
        "after_cost_per_contract_net_pnl": sum(per_contract),
        "roi": total / cost_basis if cost_basis > 0.0 else 0.0,
        "mean_sized_pnl_per_bet": statistics.fmean(pnl) if pnl else 0.0,
        "median_sized_pnl_per_bet": statistics.median(pnl) if pnl else 0.0,
        "win_count": sum(value > 0.0 for value in pnl),
        "loss_count": sum(value < 0.0 for value in pnl),
        "win_rate": sum(value > 0.0 for value in pnl) / len(pnl) if pnl else 0.0,
        "chronological_max_drawdown": max_drawdown,
        "market_bootstrap_mean_pnl_confidence_interval": ci,
        "largest_winning_market_pnl": largest_winner,
        "pnl_after_removing_largest_winner": total - largest_winner,
    }


def _group_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if field == "price_bucket":
            key = _price_bucket(float(row["execution_price"]))
        elif field == "time_to_close_bucket":
            key = _time_bucket(float(row["time_to_close_seconds"]))
        else:
            key = str(row[field])
        grouped[key].append(row)
    return {
        key: {
            "count": len(values),
            "cost_basis": sum(float(row["cost_basis"]) for row in values),
            "after_cost_sized_net_pnl": sum(
                float(row["after_cost_sized_net_pnl"]) for row in values
            ),
            "win_rate": sum(
                float(row["after_cost_sized_net_pnl"]) > 0.0 for row in values
            )
            / len(values),
        }
        for key, values in sorted(grouped.items())
    }


def _bootstrap_mean_ci(
    values: list[float], *, seed: int, resamples: int, confidence: float
) -> dict[str, float | int]:
    if not values:
        return {"lower": 0.0, "upper": 0.0, "confidence_level": confidence, "resamples": 0}
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("non-finite PnL value")
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(np.mean(generator.choice(array, size=array.size, replace=True)))
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "confidence_level": confidence,
        "resamples": resamples,
    }


def _validate_target_rows(
    rows: list[dict[str, Any]], *, expected_roles: tuple[str, ...]
) -> None:
    allowed_roles = set(expected_roles)
    if not allowed_roles:
        raise ValueError("at least one v5 source role is required")
    if not rows:
        raise ValueError(f"v5 {sorted(allowed_roles)} rows are empty")
    for row in rows:
        if str(row.get("role") or "") not in allowed_roles:
            raise ValueError("v5 source role mismatch")
        if row.get("target_used_as_decision_input") is not False:
            raise ValueError("v5 target used as decision input")
        if row.get("outcome_fields_used_as_decision_input") is not False:
            raise ValueError("v5 outcome used as decision input")
        if int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0):
            raise ValueError("v5 feature timestamp causality violation")
        target = row.get("target_net_pnl_per_contract")
        if not isinstance(target, (int, float)) or not math.isfinite(float(target)):
            raise ValueError("v5 target net PnL is missing or non-finite")


def _validate_candidate(manifest: dict[str, Any]) -> None:
    checks = {
        "candidate": manifest.get("candidate_name") == CANDIDATE_NAME,
        "actionability": manifest.get("target_free_actionability_gate_passed") is True,
        "frozen": manifest.get("research_actionability_candidate_frozen") is True,
        "future_required": manifest.get("new_strictly_later_future_holdout_required") is True,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError("v6.2 candidate manifest invalid:" + ",".join(failed))


def _validate_profile(profile: dict[str, Any]) -> None:
    checks = {
        "schema": profile.get("schema_version")
        == "bigan-v8-market-clustered-mean-ev-v6-2-v5-pnl-profile-v1",
        "target": profile.get("pnl", {}).get("target_field")
        == "target_net_pnl_per_contract",
        "market_bootstrap": profile.get("pnl", {}).get("bootstrap_unit") == "market_id",
        "retrospective": profile.get("interpretation", {}).get(
            "development_retrospective_only"
        )
        is True,
        "no_tuning": profile.get("interpretation", {}).get(
            "result_used_to_tune_v6_2_or_future_gate"
        )
        is False,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError("#213 profile invalid:" + ",".join(failed))


def _price_bucket(value: float) -> str:
    if value < 0.5:
        return "lt_0_50"
    if value < 0.6:
        return "0_50_0_60"
    if value < 0.7:
        return "0_60_0_70"
    if value <= 0.9:
        return "0_70_0_90"
    return "gt_0_90"


def _time_bucket(value: float) -> str:
    if value < 60.0:
        return "lt_60s"
    if value < 120.0:
        return "60_120s"
    if value < 180.0:
        return "120_180s"
    return "180s_plus"


def _markdown(report: dict[str, Any]) -> str:
    candidate = report["policy_metrics"]["market_clustered_mean_ev_v6_2"]
    lines = [
        "# v6.2 retrospective PnL on frozen v5 corpus",
        "",
        f"- Source markets: `{report['v5_source_total_unique_market_count']}`",
        f"- Accepted markets: `{candidate['accepted_unique_market_count']}`",
        f"- Sized after-cost PnL: `{candidate['after_cost_sized_net_pnl']:.8f}`",
        f"- Per-contract after-cost PnL: `{candidate['after_cost_per_contract_net_pnl']:.8f}`",
        f"- Cost basis: `{candidate['cost_basis']:.8f}`",
        f"- ROI: `{candidate['roi']:.6%}`",
        f"- BUY_UP: `{candidate['pnl_by_side'].get('UP', {})}`",
        f"- BUY_DOWN: `{candidate['pnl_by_side'].get('DOWN', {})}`",
        "",
        "## Interpretation",
        "",
        "This is development retrospective evidence only. The model-fit role is in-sample,",
        "and the confirmatory role supplied the v6.2 mean-risk calibration. No value in this",
        "report may tune v6.2 or the future gate. Strictly-later #212 PnL remains required.",
        "",
    ]
    return "\n".join(lines)


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.expanduser().resolve() / run_id
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"run directory already exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _result(
    run_dir: Path,
    report: dict[str, Any],
    report_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
