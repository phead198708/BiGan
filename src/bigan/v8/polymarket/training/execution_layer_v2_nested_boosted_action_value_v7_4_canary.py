"""Outcome-blind target-free canary for the frozen issue #236 v7.4 policy."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0_fit import (
    _common_feature_values,
    _side_anchor,
    validate_v7_0_training_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4 import (
    CANDIDATE_NAME,
    MODEL_SCHEMA_VERSION,
    score_nested_boosted_action_value_v7_4_market,
    validate_nested_boosted_action_value_v7_4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    _microstructure_blocking_reasons,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _blocked_safety_fields as _v6_blocked_safety_fields,
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
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

SCHEMA_PREFIX = "bigan-v8-nested-boosted-action-value-v7-4-target-free-canary"


@dataclass(frozen=True, slots=True)
class NestedBoostedActionValueV74CanaryConfig:
    """Pinned outcome-blind inputs for the one issue #236 canary."""

    run_id: str
    output_dir: Path | str
    v6_2_batch_canary_manifest_path: Path | str
    expected_v6_2_batch_canary_manifest_sha256: str
    v7_4_historical_manifest_path: Path | str
    expected_v7_4_historical_manifest_sha256: str
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(
            self.expected_v6_2_batch_canary_manifest_sha256,
            name="v6.2 batch canary manifest sha256",
        )
        _require_sha256(
            self.expected_v7_4_historical_manifest_sha256,
            name="v7.4 historical manifest sha256",
        )
        for name in (
            "output_dir",
            "v6_2_batch_canary_manifest_path",
            "v7_4_historical_manifest_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def run_nested_boosted_action_value_v7_4_target_free_canary(
    config: NestedBoostedActionValueV74CanaryConfig,
) -> dict[str, Any]:
    """Apply frozen v7.4 to one bounded batch without opening outcomes or PnL."""

    v6_2_manifest_path = config.v6_2_batch_canary_manifest_path.resolve()
    historical_manifest_path = config.v7_4_historical_manifest_path.resolve()
    _verify_pin(
        v6_2_manifest_path,
        config.expected_v6_2_batch_canary_manifest_sha256,
        "#236 v6.2 batch canary manifest",
    )
    _verify_pin(
        historical_manifest_path,
        config.expected_v7_4_historical_manifest_sha256,
        "#236 v7.4 historical manifest",
    )
    v6_2_manifest = _load_json(v6_2_manifest_path)
    historical_manifest = _load_json(historical_manifest_path)
    _validate_input_manifests(v6_2_manifest, historical_manifest)

    model_descriptor = _verified_descriptor(
        historical_manifest.get("model"), "#236 v7.4 model"
    )
    profile_descriptor = _verified_descriptor(
        historical_manifest.get("profile"), "#236 v7.4 profile"
    )
    v6_7_descriptor = _verified_descriptor(
        historical_manifest.get("v6_7_candidate_profile"), "#236 v6.7 profile"
    )
    v7_0_descriptor = _verified_descriptor(
        historical_manifest.get("v7_0_training_profile"), "#236 v7.0 profile"
    )
    model = _load_json(Path(model_descriptor["path"]))
    profile = _load_json(Path(profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    validate_nested_boosted_action_value_v7_4_profile(profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    validate_v7_0_training_profile(v7_0_profile)
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("v7_4_canary_model_schema_invalid")
    if model.get("historical_noninferiority_gate_passed") is not True:
        raise ValueError("v7_4_historical_noninferiority_gate_not_passed")

    development_descriptor = _verified_descriptor(
        v6_2_manifest.get("development_batch_canary_manifest"),
        "#236 development batch canary manifest",
    )
    v6_2_report_descriptor = _verified_descriptor(
        v6_2_manifest.get("report"), "#236 v6.2 batch canary report"
    )
    v6_2_report = _load_json(Path(v6_2_report_descriptor["path"]))
    development_manifest = _load_json(Path(development_descriptor["path"]))
    development_report_descriptor = _verified_descriptor(
        development_manifest.get("report"), "#236 development batch canary report"
    )
    development_report = _load_json(Path(development_report_descriptor["path"]))
    action_descriptor = _verified_descriptor(
        development_manifest.get("five_action_grid"), "#236 five-action grid"
    )
    scored_descriptor = _verified_descriptor(
        v6_2_manifest.get("mean_ev_scored_rows"), "#236 v6.2 scored rows"
    )
    action_rows = _load_jsonl(Path(action_descriptor["path"]))
    scored_rows = _load_jsonl(Path(scored_descriptor["path"]))
    forbidden_fields = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(_find_nonempty_fields(scored_rows, FORBIDDEN_INFERENCE_FIELDS))
    )
    if forbidden_fields:
        raise ValueError(
            "v7_4_canary_forbidden_target_fields:" + ",".join(forbidden_fields)
        )

    canonical_rows, mapping_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    decisions, replay_rows = _score_and_guard(
        canonical_rows,
        action_rows=action_rows,
        model=model,
        v6_7_profile=v6_7_profile,
    )
    accepted = [row for row in replay_rows if row["execution_guard_order_allowed"]]
    future_market_ids = sorted({str(row["market_id"]) for row in action_rows})
    fit_created_ts = int(model["fit_created_ts"])
    future_time_violations = sum(
        int(row.get("decision_ts") or 0) <= fit_created_ts
        or int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in action_rows
    )
    expected_market_count = int(
        profile["target_free_canary"]["strictly_later_outcome_blind_market_count"]
    )
    minimum_difference = int(
        profile["target_free_canary"][
            "minimum_guard_accepted_policy_difference_market_count"
        ]
    )
    accepted_difference_count = sum(
        row["execution_guard_order_allowed"]
        and row["selected_policy_decision"] != "KEEP_V6_7"
        for row in replay_rows
    )
    checks = {
        "development_data_canary_passed": development_report.get(
            "development_data_canary_passed"
        )
        is True,
        "v6_2_target_free_scoring_passed": v6_2_report.get(
            "target_free_scoring_passed"
        )
        is True
        and v6_2_manifest.get("labels_outcomes_or_pnl_opened") is False
        and v6_2_report.get("labels_outcomes_or_pnl_opened") is False,
        "exact_outcome_blind_market_count": len(future_market_ids)
        == expected_market_count,
        "strictly_later_and_causal": future_time_violations == 0,
        "complete_v7_4_decision_coverage": len(decisions)
        == len(future_market_ids),
        "canonical_sbc_mapping_complete": mapping_summary[
            "missing_scored_or_source_action_row_count"
        ]
        == 0,
        "minimum_policy_difference_support": accepted_difference_count
        >= minimum_difference,
        "forbidden_target_fields_absent": not forbidden_fields,
        "historical_noninferiority_gate_passed": model[
            "historical_noninferiority_gate_passed"
        ]
        is True,
    }
    reason_map = {
        "development_data_canary_passed": "development_data_canary_not_passed",
        "v6_2_target_free_scoring_passed": "v6_2_target_free_scoring_not_sealed",
        "exact_outcome_blind_market_count": "target_free_canary_market_count_not_exact",
        "strictly_later_and_causal": "target_free_canary_not_strictly_later_or_causal",
        "complete_v7_4_decision_coverage": "v7_4_decision_coverage_incomplete",
        "canonical_sbc_mapping_complete": "v7_4_sbc_mapping_incomplete",
        "minimum_policy_difference_support": "v7_4_policy_difference_support_below_preregistered_minimum",
        "forbidden_target_fields_absent": "forbidden_target_field_present",
        "historical_noninferiority_gate_passed": "historical_noninferiority_gate_not_passed",
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    outputs = {
        "canonical_sbc_rows": run_dir / "v7_4_target_free_canonical_sbc_rows.jsonl",
        "decisions": run_dir / "v7_4_target_free_decisions.jsonl",
        "guard_replay": run_dir / "v7_4_target_free_guard_replay.jsonl",
        "guard_accepted": run_dir / "v7_4_target_free_guard_accepted_rows.jsonl",
    }
    _write_jsonl(outputs["canonical_sbc_rows"], canonical_rows)
    _write_jsonl(outputs["decisions"], decisions)
    _write_jsonl(outputs["guard_replay"], replay_rows)
    _write_jsonl(outputs["guard_accepted"], accepted)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-report-v1",
        "run_id": config.run_id,
        "batch_id": development_report["batch_id"],
        "candidate_name": CANDIDATE_NAME,
        "historical_noninferiority_gate_passed": True,
        "model_improvement_demonstrated": historical_manifest[
            "model_improvement_demonstrated"
        ],
        "historical_equal_to_v6_7_is_allowed": True,
        "final_selected_policy_profile_name": model["final_nested_selection"][
            "selected_policy_profile_name"
        ],
        "expected_outcome_blind_market_count": expected_market_count,
        "quality_valid_market_count": len(future_market_ids),
        "candidate_scored_market_count": len(decisions),
        "canonical_sbc_row_count": len(canonical_rows),
        "mapping_summary": mapping_summary,
        "selected_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in decisions).items())
        ),
        "selected_policy_decision_distribution": dict(
            sorted(
                Counter(row["selected_policy_decision"] for row in decisions).items()
            )
        ),
        "guard_accepted_bet_count": len(accepted),
        "guard_accepted_unique_market_count": len(
            {str(row["market_id"]) for row in accepted}
        ),
        "guard_accepted_by_side": dict(
            sorted(Counter(row["selected_side"] for row in accepted).items())
        ),
        "guard_accepted_action_distribution": dict(
            sorted(Counter(row["selected_action"] for row in accepted).items())
        ),
        "guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in replay_rows
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "guard_accepted_policy_difference_market_count": accepted_difference_count,
        "minimum_guard_accepted_policy_difference_market_count": minimum_difference,
        "policy_difference_is_diagnostic_only": True,
        "target_free_canary_checks": checks,
        "target_free_canary_passed": not blockers,
        "target_free_canary_blocking_reason_codes": blockers,
        "future_strictly_later_and_disjoint_passed": future_time_violations == 0,
        "feature_timestamp_causality_violation_count": future_time_violations,
        "labels_outcomes_or_pnl_opened": False,
        "settlement_provider_called": False,
        "threshold_or_guard_tuning_performed": False,
        "source_score_mutated": False,
        "full_execution_guard_unchanged": True,
        "p_up_action_disagreement_diagnostic_only": True,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v7_4_target_free_canary_report.json"
    report_md_path = run_dir / "v7_4_target_free_canary_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _report_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-manifest-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "v6_2_batch_canary_manifest": _descriptor(v6_2_manifest_path),
        "v6_2_batch_canary_report": v6_2_report_descriptor,
        "v7_4_historical_manifest": _descriptor(historical_manifest_path),
        "development_batch_canary_manifest": development_descriptor,
        "five_action_grid": action_descriptor,
        "v6_2_scored_rows": scored_descriptor,
        "v7_4_model": model_descriptor,
        "v7_4_profile": profile_descriptor,
        "v6_7_profile": v6_7_descriptor,
        "v7_0_training_profile": v7_0_descriptor,
        **{name: _descriptor(path) for name, path in outputs.items()},
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_canary_passed": report["target_free_canary_passed"],
        "target_free_canary_blocking_reason_codes": blockers,
        "labels_outcomes_or_pnl_opened": False,
        "future_confirmatory_authorized": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_4_target_free_canary_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _canonicalize_target_free_sbc_rows(
    scored_rows: list[dict[str, Any]],
    *,
    action_rows: list[dict[str, Any]],
    v6_7_profile: dict[str, Any],
    v7_0_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_by_key = {_action_key(row): row for row in action_rows}
    scored_sbc = [row for row in scored_rows if row.get("action") in SBC_ACTIONS]
    scored_by_key = {_action_key(row): row for row in scored_sbc}
    source_sbc_keys = {
        key for key, row in source_by_key.items() if row.get("action") in SBC_ACTIONS
    }
    missing = len(set(scored_by_key) ^ source_sbc_keys)
    guard = v6_7_profile["hard_execution_safety"]
    missing_score = float(
        v7_0_profile["feature_contract"]["source_score_missing_replacement"]
    )
    status: dict[tuple[str, int, str], dict[str, Any]] = {}
    grouped_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key, scored in scored_by_key.items():
        source = source_by_key.get(key)
        if source is None:
            continue
        reasons = _microstructure_blocking_reasons(source, guard=guard)
        raw_score = _finite(scored.get("mean_ev_lower_confidence_bound"))
        available = not reasons and raw_score is not None
        score = raw_score if available else missing_score
        status[key] = {
            "available": available,
            "score": score,
            "blocking_reason_codes": reasons,
        }
        if available:
            grouped_scores[(key[0], key[1])].append(float(score))

    canonical = []
    for key in sorted(status, key=lambda item: (item[1], item[0], item[2])):
        market_id, decision_ts, action = key
        source = source_by_key[key]
        side = str(source["side"])
        item_status = status[key]
        score = float(item_status["score"])
        if item_status["available"]:
            alternatives = [
                float(other_status["score"])
                for other_key, other_status in status.items()
                if other_key[:2] == (market_id, decision_ts)
                and other_key[2] != action
                and other_status["available"]
            ]
            margin = score - max([0.0, *alternatives])
        else:
            margin = float(
                v7_0_profile["feature_contract"][
                    "source_score_margin_missing_replacement"
                ]
            )
        features = dict(source.get("decision_time_features") or {})
        micro = dict(source.get("microstructure_snapshot") or {})
        anchor = _side_anchor(
            side,
            [
                features.get("btc_return_30s"),
                features.get("btc_return_1m"),
                features.get("reference_price_to_beat_distance_at_decision"),
            ],
        )
        values = _common_feature_values(
            action_score_available=float(item_status["available"]),
            action_score=score,
            action_score_margin=margin,
            btc_anchor_direction=anchor,
            selected_side_probability=source.get("selected_side_probability"),
            execution_price=features.get("execution_price"),
            spread_bps=micro.get("spread_bps"),
            queue_fill=micro.get("queue_fill_proxy"),
            book_staleness_ms=micro.get("book_staleness_ms"),
            time_to_close_seconds=micro.get("time_to_close_seconds"),
            pre_entry_market_exposure=0.0,
            same_side_prior_entry=0.0,
            side_flip_prior_entry=0.0,
            side=side,
            profile=v7_0_profile,
        )
        row = {
            "source": "outcome_blind_v6_2_lcb_with_v6_7_execution_semantics",
            "market_id": market_id,
            "decision_group_id": f"{market_id}|{decision_ts}",
            "decision_ts": decision_ts,
            "max_input_ts": int(source["max_input_ts"]),
            "role": "target_free_canary",
            "action_family": "SELL_BEFORE_CLOSE",
            "action": action,
            "side": side,
            "decision_time_features": values,
            "source_microstructure_blocking_reason_codes": item_status[
                "blocking_reason_codes"
            ],
            "target_used_as_decision_time_input": False,
            "outcome_or_pnl_field_used_at_inference": False,
        }
        row["canonical_target_free_row_id"] = canonical_json_sha256(row)
        canonical.append(row)
    return canonical, {
        "source_scored_sbc_row_count": len(scored_sbc),
        "canonical_sbc_row_count": len(canonical),
        "missing_scored_or_source_action_row_count": missing,
        "microstructure_compatible_sbc_row_count": sum(
            item["available"] for item in status.values()
        ),
        "source_score_missing_replacement": missing_score,
        "initial_exposure_state_is_deterministic_zero": True,
    }


def _score_and_guard(
    canonical_rows: list[dict[str, Any]],
    *,
    action_rows: list[dict[str, Any]],
    model: dict[str, Any],
    v6_7_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        rows_by_market[str(row["market_id"])].append(row)
    source_by_key = {_action_key(row): row for row in action_rows}
    decisions = []
    replay = []
    for market_id in sorted(
        rows_by_market,
        key=lambda value: (
            min(int(row["decision_ts"]) for row in rows_by_market[value]),
            value,
        ),
    ):
        decision = score_nested_boosted_action_value_v7_4_market(
            rows_by_market[market_id], model_artifact=model
        )
        decisions.append(decision)
        action = str(decision["selected_action"])
        baseline_decision_ts = int(
            decision.get("baseline_decision_ts")
            or min(int(row["decision_ts"]) for row in rows_by_market[market_id])
        )
        key = (market_id, baseline_decision_ts, action)
        source = source_by_key.get(key) if action in SBC_ACTIONS else None
        if source is None:
            reasons = ["v7_4_no_positive_v6_7_baseline_action"]
            p_up_disagreement = None
        else:
            reasons = _microstructure_blocking_reasons(
                source, guard=v6_7_profile["hard_execution_safety"]
            )
            p_up_disagreement = bool(source.get("p_up_action_disagreement"))
        row = {
            **decision,
            "execution_guard_order_allowed": source is not None and not reasons,
            "execution_blocking_reason_codes": reasons,
            "p_up_action_disagreement": p_up_disagreement,
            "p_up_action_disagreement_diagnostic_only": True,
            "p_up_side_alignment_filter_enabled": False,
            "pre_entry_market_exposure": 0.0,
            "same_market_position_exists": False,
            "same_side_position_exists": False,
            "full_execution_guard_unchanged": True,
            "labels_outcomes_or_pnl_opened": False,
        }
        row["guard_replay_row_id"] = canonical_json_sha256(row)
        replay.append(row)
    return decisions, replay


def _validate_input_manifests(
    v6_2_manifest: dict[str, Any], historical_manifest: dict[str, Any]
) -> None:
    checks = {
        "v6_2_target_sealed": v6_2_manifest.get("labels_outcomes_or_pnl_opened")
        is False,
        "v7_4_candidate": historical_manifest.get("candidate_name") == CANDIDATE_NAME,
        "v7_4_historical_noninferiority": historical_manifest.get(
            "historical_noninferiority_gate_passed"
        )
        is True,
        "v7_4_canary_allowed": historical_manifest.get(
            "target_free_canary_collection_allowed"
        )
        is True,
        "v7_4_fit_leakage": historical_manifest.get("fit_leakage_audit_passed")
        is True,
    }
    for field, expected in _v6_blocked_safety_fields().items():
        checks[f"v6_2_safety_{field}"] = v6_2_manifest.get(field) == expected
    for field, expected in _v7_0_blocked_safety_fields().items():
        checks[f"v7_4_safety_{field}"] = historical_manifest.get(field) == expected
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("v7_4_canary_input_manifest_invalid:" + ",".join(failed))


def _action_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("market_id") or ""),
        int(row.get("decision_ts") or 0),
        str(row.get("action") or ""),
    )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _report_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v7.4 outcome-blind target-free canary",
            "",
            f"- Batch: `{report['batch_id']}`",
            f"- Quality-valid markets: `{report['quality_valid_market_count']}`",
            f"- Selected actions: `{report['selected_action_distribution']}`",
            f"- Guard-accepted bets: `{report['guard_accepted_bet_count']}`",
            f"- Guard-accepted sides: `{report['guard_accepted_by_side']}`",
            "- Historical gate: `non-inferior to v6.7`",
            f"- Model improvement demonstrated: `{str(report['model_improvement_demonstrated']).lower()}`",
            f"- Target-free canary passed: `{str(report['target_free_canary_passed']).lower()}`",
            f"- Blocking reasons: `{report['target_free_canary_blocking_reason_codes']}`",
            "- Outcomes/labels/PnL opened: `false`",
            "- Paper/live/promotion unlock: `false`",
            "",
        ]
    )
