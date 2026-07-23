"""Outcome-blind target-free canary for the frozen issue #245 v8.1 policy."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    _microstructure_blocking_reasons,
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_canary import (
    _action_key,
    _earliest_market_ids,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

SCHEMA_PREFIX = "bigan-v8-adaptive-support-controller-v8-1-canary"
PLAN_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-plan-v1"


@dataclass(frozen=True, slots=True)
class AdaptiveSupportControllerV81CanaryConfig:
    """Pinned inputs for the single issue #246 target-free canary."""

    run_id: str
    output_dir: Path | str
    canary_plan_path: Path | str
    expected_canary_plan_sha256: str
    development_batch_canary_manifest_path: Path | str
    expected_development_batch_canary_manifest_sha256: str
    v6_2_batch_canary_manifest_path: Path | str
    expected_v6_2_batch_canary_manifest_sha256: str
    historical_manifest_path: Path | str
    expected_historical_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_canary_plan_sha256",
            "expected_development_batch_canary_manifest_sha256",
            "expected_v6_2_batch_canary_manifest_sha256",
            "expected_historical_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "canary_plan_path",
            "development_batch_canary_manifest_path",
            "v6_2_batch_canary_manifest_path",
            "historical_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_v8_1_canary_plan(
    plan: dict[str, Any],
    *,
    historical_manifest_sha256: str,
    model_sha256: str,
    profile_sha256: str,
) -> None:
    """Reject any drift from the preregistered outcome-free canary."""

    historical = dict(plan.get("historical_gate") or {})
    scoring = dict(plan.get("frozen_scoring") or {})
    collection = dict(plan.get("collection") or {})
    gate = dict(plan.get("target_free_actionability_gate") or {})
    checks = {
        "identity": plan.get("schema_version") == PLAN_SCHEMA_VERSION
        and plan.get("issue_number") == 246
        and plan.get("candidate_name") == v81.CANDIDATE_NAME
        and plan.get("preregistered_before_collection") is True,
        "historical": historical
        == {
            "historical_manifest_sha256": historical_manifest_sha256,
            "model_sha256": model_sha256,
            "profile_sha256": profile_sha256,
            "historical_hard_gate_required": True,
            "inclusive_noninferiority_operator": "greater_than_or_equal",
            "equality_passes": True,
        },
        "scoring": scoring
        == {
            "decision_policy_implementation_commit": (
                "993801e5cc1c173a7fdf6073d9b603f84be01a19"
            ),
            "decision_policy_source_sha256": (
                "ea1e0d1a2b1252c0b1e706fe5c2f108daa3f4e3fc45f8f6afc9f97f3b409ca11"
            ),
            "v6_2_source_candidate_manifest_sha256": (
                "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9"
            ),
            "v6_7_profile_sha256": (
                "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
            ),
            "v7_0_training_profile_sha256": (
                "1f66d8699b9727651538cc34a9a2a25ba5eaac5cfded75cf8f4a258b1b5d3f4a"
            ),
            "runtime_policy_profile_sha256": (
                "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
            ),
            "fixed_edge_buffer": 0.025,
            "controller_state_advances_only_after_decision_and_full_guard_freeze": True,
            "model_score_controller_or_threshold_tuning_after_collection_allowed": False,
        },
        "collection": collection
        == {
            "mode": "bounded_candidate_agnostic_outcome_blind_raw_collection",
            "selection_method": (
                "earliest_quality_valid_strictly_later_disjoint_markets"
            ),
            "target_quality_valid_market_count": 12,
            "maximum_attempted_market_count": 18,
            "strictly_later_minimum_market_start_ts_exclusive": 1784812500000,
            "market_start_ts_must_exceed_plan_created_ts": True,
            "prior_maximum_market_close_ts": 1784689200000,
            "market_id_disjointness_required": True,
            "slug_disjointness_required": True,
            "source_row_hash_disjointness_required": True,
            "collector_protocol_sha256": (
                "2343f8247b2c1441e694b2975bccec7ae2448db5e5a5c916c3a02def49d44843"
            ),
            "feature_contract_sha256": (
                "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
            ),
            "outcomes_resolution_labels_or_pnl_opened": False,
        },
        "gate": gate
        == {
            "exact_quality_valid_market_count": 12,
            "minimum_guard_accepted_market_count": 4,
            "minimum_guard_accepted_policy_difference_market_count": 1,
            "same_decision_sbc_pair_required": True,
            "full_execution_guard_unchanged": True,
            "no_side_quota": True,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "failure_stops_before_future_pnl_collection": True,
        },
        "scope": plan.get("result_scope")
        == {
            "target_free_actionability_only": True,
            "promotion_evidence": False,
            "future_unseen_pnl_holdout_still_required": True,
        },
        "safety": plan.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#246 v8.1 canary plan invalid: " + ", ".join(blockers))


def run_adaptive_support_controller_v8_1_target_free_canary(
    config: AdaptiveSupportControllerV81CanaryConfig,
) -> dict[str, Any]:
    """Apply frozen v8.1 to the earliest 12 sealed, strictly-later markets."""

    paths = {
        "plan": config.canary_plan_path.resolve(),
        "development": config.development_batch_canary_manifest_path.resolve(),
        "v6_2": config.v6_2_batch_canary_manifest_path.resolve(),
        "historical": config.historical_manifest_path.resolve(),
    }
    pins = {
        "plan": config.expected_canary_plan_sha256,
        "development": config.expected_development_batch_canary_manifest_sha256,
        "v6_2": config.expected_v6_2_batch_canary_manifest_sha256,
        "historical": config.expected_historical_manifest_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#246 {name}")
    plan = _load_json(paths["plan"])
    development = _load_json(paths["development"])
    v6_2 = _load_json(paths["v6_2"])
    historical = _load_json(paths["historical"])
    model_descriptor = _verified_descriptor(historical.get("model"), "#246 model")
    profile_descriptor = _verified_descriptor(historical.get("profile"), "#246 profile")
    v6_7_descriptor = _verified_descriptor(
        historical.get("v6_7_candidate_profile"), "#246 v6.7 profile"
    )
    v7_0_descriptor = _verified_descriptor(
        historical.get("v7_0_training_profile"), "#246 v7.0 profile"
    )
    validate_v8_1_canary_plan(
        plan,
        historical_manifest_sha256=pins["historical"].lower(),
        model_sha256=model_descriptor["sha256"],
        profile_sha256=profile_descriptor["sha256"],
    )
    model = _load_json(Path(model_descriptor["path"]))
    profile = _load_json(Path(profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    v81.validate_adaptive_support_controller_v8_1_profile(profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    _validate_inputs(development, v6_2, historical, model)

    action_descriptor = _verified_descriptor(
        development.get("five_action_grid"), "#246 five-action grid"
    )
    scored_descriptor = _verified_descriptor(
        v6_2.get("mean_ev_scored_rows"), "#246 v6.2 scored rows"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    scored_rows = _load_jsonl(Path(scored_descriptor["path"]))
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(_find_nonempty_fields(scored_rows, FORBIDDEN_INFERENCE_FIELDS))
    )
    if forbidden:
        raise ValueError("#246 canary forbidden target fields: " + ",".join(forbidden))

    selected_markets = _earliest_market_ids(
        action_rows,
        target=int(plan["collection"]["target_quality_valid_market_count"]),
    )
    selected_set = set(selected_markets)
    action_rows = [row for row in action_rows if str(row["market_id"]) in selected_set]
    scored_rows = [row for row in scored_rows if str(row["market_id"]) in selected_set]
    candidate_rows, candidate_summary = build_v6_7_target_free_candidate_rows(
        scored_rows,
        action_rows=action_rows,
        profile=v6_7_profile,
    )
    baseline_rows = select_v6_7_target_free_rows(candidate_rows, profile=v6_7_profile)
    canonical_rows, canonical_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    decisions, guard_rows, final_state = _score_window(
        selected_markets,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=model,
        v6_7_profile=v6_7_profile,
    )

    prior_ids, prior_slugs, prior_source_hashes = _prior_identities(historical)
    selected_sbc_rows = [row for row in action_rows if row.get("action") in SBC_ACTIONS]
    selected_slugs = {str(row.get("market_slug") or "") for row in selected_sbc_rows}
    selected_source_hashes = {
        str(row.get("source_feature_row_sha256") or "") for row in selected_sbc_rows
    }
    selected_slugs.discard("")
    selected_source_hashes.discard("")
    minimum_start = max(
        int(plan["collection"]["strictly_later_minimum_market_start_ts_exclusive"]),
        int(plan["plan_created_ts"]),
    )
    causality_violations = sum(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in selected_sbc_rows
    )
    time_violations = sum(
        int(row.get("market_close_ts") or 0) - 300_000 <= minimum_start
        for row in selected_sbc_rows
    )
    market_overlap = sorted(selected_set & prior_ids)
    slug_overlap = sorted(selected_slugs & prior_slugs)
    source_hash_overlap = sorted(selected_source_hashes & prior_source_hashes)
    guard_accepted = [row for row in guard_rows if row["execution_guard_order_allowed"]]
    guard_accepted_differences = [
        row
        for row in guard_accepted
        if row["selected_action"] != row["v6_7_baseline_action"]
    ]
    gate = plan["target_free_actionability_gate"]
    checks = {
        "development_data_canary_passed": development.get(
            "development_data_canary_passed"
        )
        is True,
        "v6_2_target_free_scoring_sealed": v6_2.get(
            "labels_outcomes_or_pnl_opened"
        )
        is False,
        "historical_gate_passed": historical.get("historical_hard_gate_passed")
        is True
        and historical.get("target_free_canary_collection_allowed") is True,
        "exact_quality_valid_market_count": len(selected_markets) == 12,
        "strictly_later": time_violations == 0,
        "market_slug_and_source_disjoint": not market_overlap
        and not slug_overlap
        and not source_hash_overlap,
        "feature_timestamp_causality": causality_violations == 0,
        "complete_decision_coverage": len(decisions) == len(selected_markets),
        "canonical_sbc_mapping_complete": canonical_summary[
            "missing_scored_or_source_action_row_count"
        ]
        == 0,
        "minimum_guard_accepted_support": len(guard_accepted)
        >= int(gate["minimum_guard_accepted_market_count"]),
        "minimum_guard_accepted_policy_difference": len(guard_accepted_differences)
        >= int(gate["minimum_guard_accepted_policy_difference_market_count"]),
        "forbidden_target_fields_absent": not forbidden,
        "source_scores_unchanged": all(
            row.get("source_score_mutated") is False for row in decisions
        ),
        "controller_state_advanced_post_guard_only": all(
            row["current_guard_result_added_after_decision_freeze"] is True
            and row["current_guard_result_used_for_own_controller_decision"] is False
            for row in guard_rows
        ),
    }
    reason_map = {
        "development_data_canary_passed": "development_data_canary_not_passed",
        "v6_2_target_free_scoring_sealed": "v6_2_target_free_scoring_not_sealed",
        "historical_gate_passed": "historical_gate_not_passed",
        "exact_quality_valid_market_count": "target_free_canary_market_count_not_exact",
        "strictly_later": "target_free_canary_not_strictly_later",
        "market_slug_and_source_disjoint": "target_free_canary_identity_overlap",
        "feature_timestamp_causality": "target_free_canary_feature_causality_violation",
        "complete_decision_coverage": "target_free_canary_decision_coverage_incomplete",
        "canonical_sbc_mapping_complete": "target_free_canary_sbc_mapping_incomplete",
        "minimum_guard_accepted_support": "target_free_guard_accepted_support_insufficient",
        "minimum_guard_accepted_policy_difference": (
            "target_free_guard_accepted_policy_difference_missing"
        ),
        "forbidden_target_fields_absent": "forbidden_target_field_present",
        "source_scores_unchanged": "source_score_mutation_detected",
        "controller_state_advanced_post_guard_only": (
            "controller_state_advance_order_invalid"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    outputs = {
        "selected_market_ids": run_dir / "v8_1_target_free_selected_market_ids.json",
        "canonical_sbc_rows": run_dir / "v8_1_target_free_canonical_sbc_rows.jsonl",
        "v6_7_baseline_rows": run_dir / "v8_1_target_free_v6_7_baseline_rows.jsonl",
        "decisions": run_dir / "v8_1_target_free_decisions.jsonl",
        "guard_replay": run_dir / "v8_1_target_free_guard_replay.jsonl",
        "guard_accepted": run_dir / "v8_1_target_free_guard_accepted_rows.jsonl",
        "final_controller_state": run_dir / "v8_1_target_free_final_controller_state.json",
    }
    _write_json(outputs["selected_market_ids"], {"market_ids": selected_markets})
    _write_jsonl(outputs["canonical_sbc_rows"], canonical_rows)
    _write_jsonl(outputs["v6_7_baseline_rows"], baseline_rows)
    _write_jsonl(outputs["decisions"], decisions)
    _write_jsonl(outputs["guard_replay"], guard_rows)
    _write_jsonl(outputs["guard_accepted"], guard_accepted)
    _write_json(outputs["final_controller_state"], final_state)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "candidate_name": v81.CANDIDATE_NAME,
        "selected_market_count": len(selected_markets),
        "selected_market_ids": selected_markets,
        "v6_7_positive_guard_compatible_market_count": len(baseline_rows),
        "v6_7_candidate_summary": candidate_summary,
        "canonical_mapping_summary": canonical_summary,
        "controller_band_distribution": dict(
            sorted(
                Counter(
                    row["rank_controller_decision"]["controller_band"]
                    for row in decisions
                    if row.get("rank_controller_decision")
                ).items()
            )
        ),
        "selected_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in decisions).items())
        ),
        "guard_accepted_bet_count": len(guard_accepted),
        "guard_accepted_policy_difference_market_count": len(
            guard_accepted_differences
        ),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in guard_rows
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "effective_strictly_later_minimum_market_start_ts_exclusive": minimum_start,
        "strictly_later_time_violation_count": time_violations,
        "prior_market_overlap_count": len(market_overlap),
        "prior_slug_overlap_count": len(slug_overlap),
        "prior_source_row_hash_overlap_count": len(source_hash_overlap),
        "feature_causality_violation_count": causality_violations,
        "target_free_actionability_checks": checks,
        "target_free_canary_passed": not blockers,
        "target_free_canary_blocking_reason_codes": blockers,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "threshold_model_or_controller_tuning_performed": False,
        "source_scores_mutated": False,
        "target_free_actionability_only": True,
        "promotion_evidence": False,
        "future_unseen_pnl_holdout_still_required": True,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v8_1_target_free_canary_report.json"
    report_md_path = run_dir / "v8_1_target_free_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": v81.CANDIDATE_NAME,
        "canary_plan": _descriptor(paths["plan"]),
        "development_batch_canary_manifest": _descriptor(paths["development"]),
        "v6_2_batch_canary_manifest": _descriptor(paths["v6_2"]),
        "historical_manifest": _descriptor(paths["historical"]),
        "model": model_descriptor,
        "profile": profile_descriptor,
        "v6_7_profile": v6_7_descriptor,
        "v7_0_profile": v7_0_descriptor,
        **{name: _descriptor(path) for name, path in outputs.items()},
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_canary_passed": not blockers,
        "target_free_canary_blocking_reason_codes": blockers,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "future_unseen_pnl_holdout_authorized": not blockers,
        "future_unseen_pnl_holdout_started": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_1_target_free_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _score_window(
    selected_markets: list[str],
    *,
    canonical_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    model: dict[str, Any],
    v6_7_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical_by_key = {_action_key(row): row for row in canonical_rows}
    source_by_key = {_action_key(row): row for row in action_rows}
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    state = copy.deepcopy(model["final_rank_state"])
    decisions: list[dict[str, Any]] = []
    guard_rows: list[dict[str, Any]] = []
    for market_id in selected_markets:
        state_before = str(state["rank_state_id"])
        baseline = baseline_by_market.get(market_id)
        source = None
        if baseline is None:
            decision = _no_trade_decision(
                market_id, reason="v6_7_no_positive_guard_compatible_action"
            )
            reasons = ["v6_7_no_positive_guard_compatible_action"]
            next_state = v81.advance_adaptive_support_controller_v8_1_state(
                state, current_guard_accepted=False
            )
            baseline_action = "NO_TRADE"
        else:
            baseline_key = _action_key(baseline)
            opposite_action = (
                "BUY_DOWN_SELL_BEFORE_CLOSE"
                if baseline["action"] == "BUY_UP_SELL_BEFORE_CLOSE"
                else "BUY_UP_SELL_BEFORE_CLOSE"
            )
            opposite_key = (baseline_key[0], baseline_key[1], opposite_action)
            baseline_canonical = canonical_by_key.get(baseline_key)
            opposite_canonical = canonical_by_key.get(opposite_key)
            baseline_action = str(baseline["action"])
            if baseline_canonical is None or opposite_canonical is None:
                decision = _no_trade_decision(
                    market_id, reason="same_decision_sbc_pair_missing"
                )
                reasons = ["same_decision_sbc_pair_missing"]
                next_state = v81.advance_adaptive_support_controller_v8_1_state(
                    state, current_guard_accepted=False
                )
            else:
                decision = v81.score_adaptive_support_controller_v8_1_market(
                    {
                        "market_id": market_id,
                        "market_close_ts": int(
                            source_by_key[baseline_key]["market_close_ts"]
                        ),
                        "baseline_row": baseline_canonical,
                        "opposite_row": opposite_canonical,
                    },
                    model_artifact=model,
                    prior_rank_state=state,
                )
                selected_action = str(decision["selected_action"])
                selected_key = (baseline_key[0], baseline_key[1], selected_action)
                source = source_by_key.get(selected_key)
                reasons = (
                    ["v8_1_veto_to_no_trade"]
                    if selected_action == "NO_TRADE"
                    else ["selected_action_source_row_missing"]
                    if source is None
                    else _microstructure_blocking_reasons(
                        source, guard=v6_7_profile["hard_execution_safety"]
                    )
                )
                next_state = v81.advance_adaptive_support_controller_v8_1_state(
                    decision["next_rank_state"],
                    current_guard_accepted=source is not None and not reasons,
                )
        decision["controller_state_before_id"] = state_before
        decision["controller_state_after_id"] = next_state["rank_state_id"]
        decision["decision_id"] = canonical_json_sha256(decision)
        decisions.append(decision)
        guard_row = {
            **decision,
            "decision_ts": int(source.get("decision_ts") or 0) if source else 0,
            "v6_7_baseline_action": baseline_action,
            "execution_guard_order_allowed": source is not None and not reasons,
            "execution_blocking_reason_codes": reasons,
            "p_up_action_disagreement": (
                bool(source.get("p_up_action_disagreement")) if source else None
            ),
            "p_up_action_disagreement_diagnostic_only": True,
            "p_up_side_alignment_filter_enabled": False,
            "pre_entry_market_exposure": 0.0,
            "same_market_position_exists": False,
            "same_side_position_exists": False,
            "full_execution_guard_unchanged": True,
            "current_guard_result_used_for_own_controller_decision": False,
            "current_guard_result_added_after_decision_freeze": True,
            "labels_outcomes_or_pnl_opened": False,
        }
        guard_row["guard_replay_row_id"] = canonical_json_sha256(guard_row)
        guard_rows.append(guard_row)
        state = next_state
    return decisions, guard_rows, state


def _no_trade_decision(market_id: str, *, reason: str) -> dict[str, Any]:
    decision = {
        "market_id": market_id,
        "candidate_name": v81.CANDIDATE_NAME,
        "selected_policy_decision": "VETO_TO_NO_TRADE",
        "selected_action": "NO_TRADE",
        "selected_side": "NONE",
        "trade_selected": False,
        "selection_reason_codes": [reason],
        "rank_controller_decision": None,
        "source_score_mutated": False,
        "outcome_or_pnl_field_used_at_inference": False,
        **_v7_0_blocked_safety_fields(),
    }
    decision["decision_id"] = canonical_json_sha256(decision)
    return decision


def _prior_identities(
    historical: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    market_ids: set[str] = set()
    slugs: set[str] = set()
    source_hashes: set[str] = set()
    for key in ("seed_runtime_target_rows", "consumed_stream_five_action_rows"):
        descriptor = _verified_descriptor(historical.get(key), f"#246 prior {key}")
        for row in _load_jsonl(Path(descriptor["path"])):
            market_ids.add(str(row["market_id"]))
            if row.get("market_slug"):
                slugs.add(str(row["market_slug"]))
            if row.get("source_feature_row_sha256"):
                source_hashes.add(str(row["source_feature_row_sha256"]))
    return market_ids, slugs, source_hashes


def _validate_inputs(
    development: dict[str, Any],
    v6_2: dict[str, Any],
    historical: dict[str, Any],
    model: dict[str, Any],
) -> None:
    checks = {
        "development": development.get("development_data_canary_passed") is True
        and development.get("labels_outcomes_or_pnl_opened") is False,
        "v6_2": v6_2.get("labels_outcomes_or_pnl_opened") is False,
        "historical": historical.get("candidate_name") == v81.CANDIDATE_NAME
        and historical.get("historical_hard_gate_passed") is True
        and historical.get("target_free_canary_collection_allowed") is True,
        "model": model.get("schema_version") == v81.MODEL_SCHEMA_VERSION
        and model.get("historical_hard_gate_passed") is True
        and model.get("target_free_canary_collection_allowed") is True,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#246 canary input invalid: " + ",".join(blockers))


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v8.1 target-free canary",
            "",
            f"- run: `{report['run_id']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- controller bands: `{report['controller_band_distribution']}`",
            f"- selected actions: `{report['selected_action_distribution']}`",
            f"- guard-accepted bets: `{report['guard_accepted_bet_count']}`",
            "- guard-accepted policy differences: "
            f"`{report['guard_accepted_policy_difference_market_count']}`",
            "- target-free canary passed: "
            f"`{str(report['target_free_canary_passed']).lower()}`",
            f"- blockers: `{report['target_free_canary_blocking_reason_codes']}`",
            "- labels/outcomes/resolution/PnL opened: `false`",
            "- promotion evidence: `false`",
            "- future unseen PnL holdout still required: `true`",
            "- paper/live/write/wallet/capital/handoff remain blocked.",
            "",
        ]
    )
