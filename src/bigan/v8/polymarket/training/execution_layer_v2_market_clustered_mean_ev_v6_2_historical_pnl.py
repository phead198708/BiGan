"""Outcome-aware historical PnL diagnostic for the frozen v6.2 candidate."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xgboost as xgb

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_conformal_net_return_v5 import (
    _raw_target_stripped_predictions,
    apply_conformal_scores,
)
from bigan.v8.polymarket.training.execution_layer_v2_guard_compatible_direct_net_return_v4 import (
    TARGET_FIELDS,
    _row_key,
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
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6_fit import (
    attach_frozen_execution_compatibility,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_on_v5_target_free_diagnostic import (
    _normalize_v5_labeled_rows,
)

SCHEMA_PREFIX = "bigan-v8-market-clustered-mean-ev-v6-2-historical-pnl"
MODEL_FIT_ROLE = "historical_model_fit"
RISK_CALIBRATION_ROLE = "historical_mean_risk_calibration"


@dataclass(frozen=True, slots=True)
class MarketClusteredMeanEVV62HistoricalPnlConfig:
    """Pinned inputs for one non-promotional historical PnL diagnostic."""

    run_id: str
    output_dir: Path | str
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
        for name in (
            "expected_candidate_manifest_sha256",
            "expected_v5_freeze_manifest_sha256",
            "expected_feature_contract_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        _require_git_sha(self.implementation_commit)
        for name in (
            "output_dir",
            "candidate_manifest_path",
            "v5_freeze_manifest_path",
            "feature_contract_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_market_clustered_mean_ev_v6_2_historical_pnl(
    config: MarketClusteredMeanEVV62HistoricalPnlConfig,
) -> dict[str, Any]:
    """Replay frozen v6.2 and matched v5 before joining historical targets."""

    candidate_path = config.candidate_manifest_path.resolve()
    v5_path = config.v5_freeze_manifest_path.resolve()
    feature_contract_path = config.feature_contract_path.resolve()
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
    candidate = _load_json(candidate_path)
    v5 = _load_json(v5_path)
    feature_contract = _load_json(feature_contract_path)
    descriptors = _validate_lineage(
        candidate,
        v5=v5,
        feature_contract_sha256=config.expected_feature_contract_sha256,
    )
    feature_columns = tuple(str(value) for value in feature_contract["feature_columns"])

    model_fit_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(descriptors["model_fit_rows"]["path"])),
        role=MODEL_FIT_ROLE,
        expected_source_roles={"development_train", "development_calibration"},
        feature_columns=feature_columns,
    )
    risk_calibration_rows = _normalize_v5_labeled_rows(
        _load_jsonl(Path(descriptors["risk_calibration_rows"]["path"])),
        role=RISK_CALIBRATION_ROLE,
        expected_source_roles={"confirmatory_validation"},
        feature_columns=feature_columns,
    )
    _validate_role_counts(model_fit_rows, risk_calibration_rows)
    labeled_rows = [*model_fit_rows, *risk_calibration_rows]
    targets = {
        _row_key(row): {
            "target_net_pnl_per_contract": float(row["target_net_pnl_per_contract"]),
            "historical_source_role": str(row["source_v5_role"]),
            "historical_model_usage_role": str(row["role"]),
        }
        for row in labeled_rows
    }
    if len(targets) != len(labeled_rows):
        raise ValueError("historical action target identity is not unique")

    booster = xgb.Booster()
    booster.load_model(descriptors["model"]["path"])
    raw_predictions = attach_frozen_execution_compatibility(
        _raw_target_stripped_predictions(
            booster,
            labeled_rows,
            feature_columns=feature_columns,
        )
    )
    if _contains_target_fields(raw_predictions):
        raise ValueError("historical target entered target-free prediction rows")

    candidate_scored = apply_market_clustered_mean_ev_scores(
        raw_predictions,
        calibration_artifact=_load_json(Path(descriptors["v6_2_calibration"]["path"])),
    )
    candidate_replay = _outcome_blind_acceptance_replay(
        candidate_scored,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    baseline_scored = _attach_matched_v5_replay_fields(
        apply_conformal_scores(
            raw_predictions,
            calibration_artifact=_load_json(Path(descriptors["v5_calibration"]["path"])),
            profile=_load_json(Path(descriptors["v5_profile"]["path"])),
        )
    )
    baseline_replay = _outcome_blind_acceptance_replay(
        baseline_scored,
        entry_threshold=0.0,
        runner_up_advantage_threshold=0.0,
    )
    if _contains_target_fields(candidate_replay + baseline_replay):
        raise ValueError("historical target entered outcome-blind guard replay")

    candidate_evaluation = join_historical_targets_after_replay(
        candidate_replay,
        targets=targets,
        policy_name=CANDIDATE_NAME,
    )
    baseline_evaluation = join_historical_targets_after_replay(
        baseline_replay,
        targets=targets,
        policy_name="matched_v5_individual_outcome_conformal",
    )
    report = build_historical_pnl_report(
        config=config,
        candidate_rows=candidate_evaluation,
        baseline_rows=baseline_evaluation,
        labeled_rows=labeled_rows,
    )

    run_dir = config.output_dir.resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run directory exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    candidate_replay_path = run_dir / "v6_2_historical_target_free_guard_replay.jsonl"
    baseline_replay_path = run_dir / "matched_v5_historical_target_free_guard_replay.jsonl"
    candidate_eval_path = run_dir / "v6_2_historical_settled_evaluation_rows.jsonl"
    baseline_eval_path = run_dir / "matched_v5_historical_settled_evaluation_rows.jsonl"
    report_path = run_dir / "v6_2_on_v5_historical_pnl_report.json"
    report_md_path = run_dir / "v6_2_on_v5_historical_pnl_report.md"
    _write_jsonl(candidate_replay_path, candidate_replay)
    _write_jsonl(baseline_replay_path, baseline_replay)
    _write_jsonl(candidate_eval_path, candidate_evaluation)
    _write_jsonl(baseline_eval_path, baseline_evaluation)
    _write_json(report_path, report)
    _write_text(report_md_path, _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "candidate_manifest": _descriptor(candidate_path),
        "v5_freeze_manifest": _descriptor(v5_path),
        "feature_contract": _descriptor(feature_contract_path),
        "source_model": descriptors["model"],
        "source_v6_2_calibration": descriptors["v6_2_calibration"],
        "source_v5_calibration": descriptors["v5_calibration"],
        "source_model_fit_rows": descriptors["model_fit_rows"],
        "source_risk_calibration_rows": descriptors["risk_calibration_rows"],
        "candidate_target_free_guard_replay": _descriptor(candidate_replay_path),
        "matched_v5_target_free_guard_replay": _descriptor(baseline_replay_path),
        "candidate_historical_evaluation_rows": _descriptor(candidate_eval_path),
        "matched_v5_historical_evaluation_rows": _descriptor(baseline_eval_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "historical_outcome_aware_diagnostic_only": True,
        "promotion_evidence": False,
        "future_side_only_gate_unchanged": True,
        **_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_2_on_v5_historical_pnl_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def join_historical_targets_after_replay(
    replay_rows: list[dict[str, Any]],
    *,
    targets: dict[tuple[str, int, str], dict[str, Any]],
    policy_name: str,
) -> list[dict[str, Any]]:
    """Join an executed action target only after target-free guard replay is frozen."""

    if _contains_target_fields(replay_rows):
        raise ValueError("target field present before historical replay target join")
    output = []
    for replay in replay_rows:
        action = str(replay["executed_action"])
        key = (str(replay["market_id"]), int(replay["decision_ts"]), action)
        target = targets.get(key)
        if target is None:
            raise ValueError("historical replay action target identity is missing")
        allowed = replay.get("execution_guard_order_allowed") is True
        order_size = float(replay.get("proposed_order_size") or 0.0)
        if allowed and order_size <= 0.0:
            raise ValueError("guard-allowed historical row has invalid order size")
        target_value = float(target["target_net_pnl_per_contract"])
        row = {
            **replay,
            "policy_name": policy_name,
            "historical_source_role": target["historical_source_role"],
            "historical_model_usage_role": target["historical_model_usage_role"],
            "target_net_pnl_per_contract": target_value,
            "accepted_bet_net_pnl": order_size * target_value if allowed else 0.0,
            "target_joined_after_outcome_blind_guard_replay": True,
            "target_used_as_decision_input": False,
            "outcome_fields_used_as_decision_input": False,
            "historical_outcomes_used_for_tuning": False,
            **_blocked_safety_fields(),
        }
        row["historical_evaluation_row_sha256"] = canonical_json_sha256(row)
        output.append(row)
    return output


def build_historical_pnl_report(
    *,
    config: MarketClusteredMeanEVV62HistoricalPnlConfig,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    labeled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build split-aware descriptive metrics without an eligibility decision."""

    role_groups = {
        "development_train_90": {"development_train"},
        "development_calibration_45": {"development_calibration"},
        "model_fit_135": {"development_train", "development_calibration"},
        "mean_risk_calibration_60": {"confirmatory_validation"},
        "combined_frozen_v5_195": {
            "development_train",
            "development_calibration",
            "confirmatory_validation",
        },
    }
    candidate = {
        name: historical_policy_metrics(
            [row for row in candidate_rows if row["historical_source_role"] in roles]
        )
        for name, roles in role_groups.items()
    }
    baseline = {
        name: historical_policy_metrics(
            [row for row in baseline_rows if row["historical_source_role"] in roles]
        )
        for name, roles in role_groups.items()
    }
    combined_name = "combined_frozen_v5_195"
    combined_candidate = candidate[combined_name]
    combined_baseline = baseline[combined_name]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "report_id": None,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "historical_market_count": len({str(row["market_id"]) for row in labeled_rows}),
        "historical_action_row_count": len(labeled_rows),
        "source_role_market_count": dict(
            sorted(
                Counter(
                    str(row["source_v5_role"])
                    for row in {str(row["market_id"]): row for row in labeled_rows}.values()
                ).items()
            )
        ),
        "candidate_metrics_by_split": candidate,
        "matched_v5_metrics_by_split": baseline,
        "final_combined_candidate_post_cost_net_pnl": combined_candidate[
            "accepted_bet_net_pnl_sum"
        ],
        "final_combined_matched_v5_post_cost_net_pnl": combined_baseline[
            "accepted_bet_net_pnl_sum"
        ],
        "final_combined_candidate_minus_matched_v5_post_cost_net_pnl": (
            combined_candidate["accepted_bet_net_pnl_sum"]
            - combined_baseline["accepted_bet_net_pnl_sum"]
        ),
        "primary_pnl_aggregation": "buy_up_buy_down_side_only",
        "action_and_family_metrics_diagnostic_only": True,
        "point_model_fit_split_is_in_sample": True,
        "mean_risk_calibration_split_used_for_calibration": True,
        "no_strictly_unseen_split_in_this_report": True,
        "historical_outcome_aware_diagnostic_only": True,
        "promotion_evidence": False,
        "uses_historical_pnl_for_tuning": False,
        "future_side_only_gate_unchanged": True,
        "current_or_future_holdout_outcomes_opened": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def historical_policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate chronological, side-first metrics for one descriptive split."""

    accepted = [row for row in rows if row["execution_guard_order_allowed"] is True]
    accepted.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["executed_action"]),
        )
    )
    market_pnl: dict[str, float] = defaultdict(float)
    for row in accepted:
        market_pnl[str(row["market_id"])] += float(row["accepted_bet_net_pnl"])
    pnl = sum(market_pnl.values())
    largest_winner = max(market_pnl.values(), default=0.0)
    cost_basis = sum(_row_cost_basis(row) for row in accepted)
    sides: dict[str, list[dict[str, Any]]] = defaultdict(list)
    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        sides[str(row["selected_side"])].append(row)
        action = str(row["executed_action"])
        actions[action].append(row)
        families[_action_family(action)].append(row)
    return {
        "decision_count": len(rows),
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(market_pnl),
        "accepted_side_distribution": dict(
            sorted(Counter(str(row["selected_side"]) for row in accepted).items())
        ),
        "accepted_action_distribution": dict(
            sorted(Counter(str(row["executed_action"]) for row in accepted).items())
        ),
        "accepted_family_distribution": dict(
            sorted(Counter(_action_family(str(row["executed_action"])) for row in accepted).items())
        ),
        "accepted_bet_cost_basis": cost_basis,
        "accepted_bet_net_pnl_sum": pnl,
        "accepted_bet_net_pnl_mean": pnl / len(accepted) if accepted else 0.0,
        "accepted_bet_roi": pnl / cost_basis if cost_basis > 0.0 else 0.0,
        "win_rate": (
            sum(float(row["accepted_bet_net_pnl"]) > 0.0 for row in accepted)
            / len(accepted)
            if accepted
            else 0.0
        ),
        "chronological_max_drawdown": _chronological_max_drawdown(accepted),
        "largest_winning_market_pnl": largest_winner,
        "largest_winner_removed_pnl": pnl - max(largest_winner, 0.0),
        "pnl_by_side": {
            side: _group_metrics(group) for side, group in sorted(sides.items())
        },
        "pnl_by_action": {
            action: _group_metrics(group) for action, group in sorted(actions.items())
        },
        "pnl_by_action_family": {
            family: _group_metrics(group) for family, group in sorted(families.items())
        },
    }


def _validate_lineage(
    candidate: dict[str, Any],
    *,
    v5: dict[str, Any],
    feature_contract_sha256: str,
) -> dict[str, dict[str, str]]:
    if candidate.get("candidate_name") != CANDIDATE_NAME:
        raise ValueError("candidate name mismatch")
    if candidate.get("research_actionability_candidate_frozen") is not True:
        raise ValueError("v6.2 candidate is not frozen")
    if candidate.get("target_free_actionability_gate_passed") is not True:
        raise ValueError("v6.2 actionability gate did not pass")
    model = _verified_descriptor(candidate.get("source_model"), "v6.2 source model")
    v6_2_calibration = _verified_descriptor(
        candidate.get("market_clustered_mean_risk_calibration"),
        "v6.2 mean-risk calibration",
    )
    v5_model = _verified_descriptor(v5.get("model"), "v5 source model")
    v5_calibration = _verified_descriptor(v5.get("calibration_artifact"), "v5 calibration")
    v5_profile = _verified_descriptor(v5.get("fit_profile"), "v5 profile")
    model_fit_rows = _verified_descriptor(
        v5.get("development_train_action_rows"), "v5 model-fit action rows"
    )
    risk_calibration_rows = _verified_descriptor(
        v5.get("development_calibration_action_rows"), "v5 calibration action rows"
    )
    if model["sha256"] != v5_model["sha256"]:
        raise ValueError("v6.2 and matched v5 model lineage differs")
    if candidate.get("source_v5_conformal_action_rows") != risk_calibration_rows:
        raise ValueError("v6.2 risk calibration row lineage differs from v5")
    for artifact, name in (
        (_load_json(Path(v6_2_calibration["path"])), "v6.2 calibration"),
        (_load_json(Path(v5_calibration["path"])), "v5 calibration"),
    ):
        observed = artifact.get("feature_contract_sha256")
        if observed is not None and observed != feature_contract_sha256:
            raise ValueError(f"{name} feature contract mismatch")
    for key, expected in _blocked_safety_fields().items():
        if candidate.get(key) != expected or v5.get(key) != expected:
            raise ValueError(f"frozen lineage safety mismatch: {key}")
    return {
        "model": model,
        "v6_2_calibration": v6_2_calibration,
        "v5_calibration": v5_calibration,
        "v5_profile": v5_profile,
        "model_fit_rows": model_fit_rows,
        "risk_calibration_rows": risk_calibration_rows,
    }


def _validate_role_counts(
    model_fit_rows: list[dict[str, Any]],
    risk_calibration_rows: list[dict[str, Any]],
) -> None:
    fit_markets = {str(row["market_id"]) for row in model_fit_rows}
    calibration_markets = {str(row["market_id"]) for row in risk_calibration_rows}
    if len(fit_markets) != 135 or len(calibration_markets) != 60:
        raise ValueError("historical v5 role market counts are not 135/60")
    if fit_markets & calibration_markets:
        raise ValueError("historical v5 fit and calibration markets overlap")


def _attach_matched_v5_replay_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        raw = float(row["raw_direct_predicted_net_return"])
        output.append(
            {
                **row,
                "raw_pairwise_rank_score": raw,
                "pairwise_group_normalized_rank_score": raw,
                "action_advantage_lcb_score_bucket": "matched_v5_historical_diagnostic",
                "action_advantage_lcb_estimate_source": row["ranking_score_source"],
            }
        )
    return output


def _contains_target_fields(rows: list[dict[str, Any]]) -> bool:
    return any(any(_nonempty(row.get(field)) for field in TARGET_FIELDS) for row in rows)


def _nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _row_cost_basis(row: dict[str, Any]) -> float:
    size = float(row.get("proposed_order_size") or 0.0)
    snapshot = dict(row.get("microstructure_snapshot") or {})
    features = dict(row.get("decision_time_features") or {})
    price = float(snapshot.get("entry_ask") or features.get("execution_price") or 0.0)
    return size * price


def _chronological_max_drawdown(rows: list[dict[str, Any]]) -> float:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    for row in rows:
        cumulative += float(row["accepted_bet_net_pnl"])
        peak = max(peak, cumulative)
        maximum = max(maximum, peak - cumulative)
    return maximum


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = sum(float(row["accepted_bet_net_pnl"]) for row in rows)
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "accepted_bet_net_pnl_sum": pnl,
        "accepted_bet_net_pnl_mean": pnl / len(rows) if rows else 0.0,
        "win_rate": (
            sum(float(row["accepted_bet_net_pnl"]) > 0.0 for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def _action_family(action: str) -> str:
    if action.endswith("SELL_BEFORE_CLOSE"):
        return "SELL_BEFORE_CLOSE"
    if action.endswith("HOLD_TO_SETTLEMENT"):
        return "HOLD_TO_SETTLEMENT"
    return "NO_TRADE"


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v6.2 on frozen v5 historical PnL",
        "",
        "This is outcome-aware historical diagnostic evidence only. It is not promotion evidence.",
        "",
        "| Split | Candidate bets | Candidate PnL | Matched v5 bets | Matched v5 PnL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["candidate_metrics_by_split"].items():
        baseline = report["matched_v5_metrics_by_split"][name]
        lines.append(
            f"| {name} | {metrics['guard_accepted_bet_count']} | "
            f"{metrics['accepted_bet_net_pnl_sum']:.9f} | "
            f"{baseline['guard_accepted_bet_count']} | "
            f"{baseline['accepted_bet_net_pnl_sum']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Model-fit 135 is in-sample for the point model.",
            "- Mean-risk calibration 60 was used to calibrate v6.2.",
            "- No split in this report is a strictly unseen future evaluation.",
            "- BUY_UP/BUY_DOWN is primary; action/family breakdowns are diagnostic only.",
            "- The exact-200 future side-only gate remains unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"
