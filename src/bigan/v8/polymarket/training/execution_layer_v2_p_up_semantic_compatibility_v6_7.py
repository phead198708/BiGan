"""Freeze the #227 p_up-semantic execution-compatible v6.7 candidate."""

from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _blocked_safety_fields,
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _sha256_file,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)

PROFILE_SCHEMA_VERSION = "bigan-v8-p-up-semantic-execution-compatibility-v6-7-profile-v1"
ROW_SCHEMA_VERSION = "bigan-v8-p-up-semantic-execution-compatibility-v6-7-row-v1"
REPORT_SCHEMA_VERSION = "bigan-v8-p-up-semantic-execution-compatibility-v6-7-report-v1"
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-p-up-semantic-execution-compatibility-v6-7-freeze-manifest-v1"
)
CANDIDATE_NAME = "p_up_semantic_execution_compatibility_v6_7"
SBC_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}
SIDES = ("UP", "DOWN")
FORBIDDEN_TARGET_FIELDS = {
    "resolved_outcome",
    "winning_outcome",
    "settlement_pnl",
    "realized_trade_pnl",
    "total_polymarket_pnl",
    "runtime_policy_after_cost_net_pnl_per_contract",
    "future_return",
    "label",
}


@dataclass(frozen=True, slots=True)
class PUpSemanticCompatibilityV67Config:
    """Pinned inputs for one outcome-free #227 candidate freeze."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    source_freeze_manifest_path: Path | str
    expected_source_freeze_manifest_sha256: str
    predictions_path: Path | str
    expected_predictions_sha256: str
    five_action_rows_path: Path | str
    expected_five_action_rows_sha256: str
    legacy_guard_replay_path: Path | str
    expected_legacy_guard_replay_sha256: str
    implementation_commit: str
    candidate_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        for name in (
            "expected_profile_sha256",
            "expected_source_freeze_manifest_sha256",
            "expected_predictions_sha256",
            "expected_five_action_rows_sha256",
            "expected_legacy_guard_replay_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        _require_git_sha(self.implementation_commit)
        if self.candidate_freeze_created_ts <= 0:
            raise ValueError("candidate_freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "source_freeze_manifest_path",
            "predictions_path",
            "five_action_rows_path",
            "legacy_guard_replay_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_p_up_semantic_compatibility_v6_7_profile(
    profile: dict[str, Any],
) -> None:
    """Reject any drift from the preregistered #227 semantic-only change."""

    lineage = dict(profile.get("source_lineage") or {})
    selection = dict(profile.get("selection_rule") or {})
    guard = dict(profile.get("hard_execution_safety") or {})
    semantics = dict(profile.get("p_up_semantics") or {})
    support = dict(profile.get("target_free_support_gate") or {})
    prohibited = dict(profile.get("prohibited") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 227,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "lineage": set(lineage)
        == {
            "source_freeze_manifest_sha256",
            "v6_2_target_free_predictions_sha256",
            "target_free_five_action_rows_sha256",
            "legacy_guard_replay_sha256",
            "v6_2_source_model_sha256",
            "v6_2_calibration_sha256",
        }
        and all(_is_sha256(str(value)) for value in lineage.values()),
        "selection": selection
        == {
            "eligible_actions": sorted(SBC_ACTIONS),
            "ranking_score_field": "mean_ev_lower_confidence_bound",
            "ranking_score_source": "frozen_v6_2_market_clustered_mean_ev_lcb",
            "entry_threshold": 0.0,
            "threshold_operator": "strictly_greater_than",
            "one_row_per_market": True,
            "tie_break": [
                "highest_mean_ev_lower_confidence_bound",
                "earliest_decision_ts",
                "lexicographic_action",
            ],
            "target_fields_used_for_selection": False,
            "score_mutation_allowed": False,
        },
        "guard": guard
        == {
            "max_spread_bps": 1000.0,
            "max_book_staleness_ms": 2000.0,
            "min_queue_fill_probability_proxy": 0.5,
            "min_time_to_close_seconds": 60.0,
            "positive_execution_price_required": True,
            "positive_executable_ask_notional_required": True,
            "positive_executable_bid_notional_required": True,
            "positive_liquidity_depth_required": True,
            "feature_causality_required": True,
            "reference_provenance_required": True,
            "exposure_duplicate_position_and_sizing_guards_unchanged": True,
        },
        "semantics": semantics
        == {
            "p_up_side_alignment_filter_enabled": False,
            "p_up_action_disagreement_diagnostic_only": True,
            "market_implied_probability_used_as_direct_fair_value_ev": False,
            "market_implied_probability_used_as_conditioning_feature": True,
            "market_implied_probability_used_as_regime_direction_vote": False,
            "p_up_disagreement_removed_from_execution_safety_blockers_only": True,
        },
        "support": support
        == {
            "expected_market_count": 60,
            "minimum_unique_selected_market_count_per_side": 20,
            "require_both_sides": True,
            "labels_outcomes_settlement_or_pnl_access_allowed": False,
            "result_selected_rerun_allowed": False,
        },
        "prohibited": prohibited
        == {
            "source_score_mutation_allowed": False,
            "execution_safety_threshold_tuning_allowed": False,
            "validation_or_future_labels_used_for_selection": False,
            "settlement_outcome_or_pnl_access_before_freeze": False,
            "paper_live_write_wallet_or_capital_unlock_allowed": False,
        },
        "safety": profile.get("safety") == _blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#227 profile invalid: " + ", ".join(blockers))


def run_p_up_semantic_compatibility_v6_7(
    config: PUpSemanticCompatibilityV67Config,
) -> dict[str, Any]:
    """Freeze the semantic adapter and evaluate the sealed target-free canary."""

    paths = {
        "profile": Path(config.profile_path).resolve(),
        "source_freeze_manifest": Path(config.source_freeze_manifest_path).resolve(),
        "predictions": Path(config.predictions_path).resolve(),
        "five_action_rows": Path(config.five_action_rows_path).resolve(),
        "legacy_guard_replay": Path(config.legacy_guard_replay_path).resolve(),
    }
    expected = {
        "profile": config.expected_profile_sha256,
        "source_freeze_manifest": config.expected_source_freeze_manifest_sha256,
        "predictions": config.expected_predictions_sha256,
        "five_action_rows": config.expected_five_action_rows_sha256,
        "legacy_guard_replay": config.expected_legacy_guard_replay_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, expected[name], f"#227 {name}")

    profile = _load_json(paths["profile"])
    validate_p_up_semantic_compatibility_v6_7_profile(profile)
    lineage = dict(profile["source_lineage"])
    lineage_checks = {
        "source_freeze_manifest": expected["source_freeze_manifest"]
        == lineage["source_freeze_manifest_sha256"],
        "predictions": expected["predictions"]
        == lineage["v6_2_target_free_predictions_sha256"],
        "five_action_rows": expected["five_action_rows"]
        == lineage["target_free_five_action_rows_sha256"],
        "legacy_guard_replay": expected["legacy_guard_replay"]
        == lineage["legacy_guard_replay_sha256"],
    }
    if not all(lineage_checks.values()):
        raise ValueError("#227 profile lineage does not match pinned inputs")

    source_manifest = _load_json(paths["source_freeze_manifest"])
    _validate_source_freeze_manifest(source_manifest, expected=expected)
    predictions = _load_jsonl(paths["predictions"])
    action_rows = _load_jsonl(paths["five_action_rows"])
    legacy_replay = _load_jsonl(paths["legacy_guard_replay"])
    _validate_target_free_inputs(predictions, action_rows, legacy_replay)

    candidate_rows, candidate_summary = build_v6_7_target_free_candidate_rows(
        predictions,
        action_rows=action_rows,
        profile=profile,
    )
    selected_rows = select_v6_7_target_free_rows(candidate_rows, profile=profile)
    selected_side_count = Counter(str(row["side"]) for row in selected_rows)
    selected_market_count = len({str(row["market_id"]) for row in selected_rows})
    maximum_selected_decision_ts = max(
        int(row["decision_ts"]) for row in selected_rows
    )
    if config.candidate_freeze_created_ts <= maximum_selected_decision_ts:
        raise ValueError("#227 candidate freeze timestamp is not after source decisions")
    minimum_side = int(
        profile["target_free_support_gate"][
            "minimum_unique_selected_market_count_per_side"
        ]
    )
    support_checks = {
        "exact_market_count": selected_market_count
        == int(profile["target_free_support_gate"]["expected_market_count"]),
        "one_row_per_market": selected_market_count == len(selected_rows),
        "buy_up_support": selected_side_count["UP"] >= minimum_side,
        "buy_down_support": selected_side_count["DOWN"] >= minimum_side,
        "all_selected_scores_positive": all(
            float(row["v6_7_selection_score"]) > 0.0 for row in selected_rows
        ),
        "all_selected_rows_microstructure_safe": all(
            row["microstructure_safety_passed"] is True for row in selected_rows
        ),
        "feature_causality": all(
            int(row["max_input_ts"]) <= int(row["decision_ts"])
            for row in selected_rows
        ),
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in selected_rows
        ),
        "targets_remained_sealed": True,
    }
    blockers = [
        f"{name}_gate_failed" for name, passed in support_checks.items() if not passed
    ]
    support_passed = not blockers
    legacy_summary = _legacy_guard_summary(legacy_replay)

    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run path exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    candidate_path = run_dir / "v6_7_p_up_semantic_candidate_rows.jsonl"
    selected_path = run_dir / "v6_7_p_up_semantic_selected_rows.jsonl"
    report_path = run_dir / "v6_7_p_up_semantic_attrition_report.json"
    markdown_path = run_dir / "v6_7_p_up_semantic_attrition_report.md"
    manifest_path = run_dir / "v6_7_p_up_semantic_candidate_freeze_manifest.json"
    _write_jsonl(candidate_path, candidate_rows)
    _write_jsonl(selected_path, selected_rows)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "issue_number": 227,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": config.candidate_freeze_created_ts,
        "maximum_selected_decision_ts": maximum_selected_decision_ts,
        "profile_sha256": expected["profile"],
        "source_freeze_manifest_sha256": expected["source_freeze_manifest"],
        "source_prediction_sha256": expected["predictions"],
        "source_five_action_rows_sha256": expected["five_action_rows"],
        "source_legacy_guard_replay_sha256": expected["legacy_guard_replay"],
        "source_lineage_hashes_verified": all(lineage_checks.values()),
        "source_market_count": len(
            {str(row["market_id"]) for row in action_rows}
        ),
        "source_prediction_row_count": len(predictions),
        "source_five_action_row_count": len(action_rows),
        "candidate_summary": candidate_summary,
        "legacy_guard_summary": legacy_summary,
        "v6_7_selected_market_count": selected_market_count,
        "v6_7_selected_side_count": dict(sorted(selected_side_count.items())),
        "v6_7_selected_p_up_disagreement_count": sum(
            row["p_up_action_disagreement"] is True for row in selected_rows
        ),
        "v6_7_selected_p_up_agreement_count": sum(
            row["p_up_action_disagreement"] is False for row in selected_rows
        ),
        "target_free_support_gate_checks": support_checks,
        "target_free_support_gate_passed": support_passed,
        "target_free_support_blocking_reason_codes": blockers,
        "p_up_side_alignment_filter_enabled": False,
        "p_up_action_disagreement_diagnostic_only": True,
        "p_up_removed_from_execution_safety_blockers_only": True,
        "hard_execution_safety_thresholds_unchanged": True,
        "exposure_duplicate_position_and_sizing_guards_unchanged": True,
        "market_implied_probability_used_as_direct_fair_value_ev": False,
        "market_implied_probability_used_as_conditioning_feature": True,
        "market_implied_probability_used_as_regime_direction_vote": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "target_fields_used_for_selection": False,
        "validation_or_future_labels_used_for_tuning": False,
        "source_score_mutated": False,
        "candidate_scoring_frozen": support_passed,
        "strictly_later_outcome_blind_collection_allowed": support_passed,
        "future_target_access_allowed": False,
        **_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    _write_json(report_path, report)
    _write_text(markdown_path, _report_markdown(report))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "issue_number": 227,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "candidate_freeze_created_ts": config.candidate_freeze_created_ts,
        "future_collection_minimum_created_ts_exclusive": (
            config.candidate_freeze_created_ts
        ),
        "profile": _descriptor(paths["profile"]),
        "source_freeze_manifest": _descriptor(paths["source_freeze_manifest"]),
        "source_target_free_predictions": _descriptor(paths["predictions"]),
        "source_target_free_five_action_rows": _descriptor(
            paths["five_action_rows"]
        ),
        "source_legacy_guard_replay": _descriptor(paths["legacy_guard_replay"]),
        "candidate_rows": _descriptor(candidate_path),
        "selected_rows": _descriptor(selected_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(markdown_path),
        "candidate_scoring_frozen": support_passed,
        "target_free_support_gate_passed": support_passed,
        "target_free_support_blocking_reason_codes": blockers,
        "strictly_later_outcome_blind_collection_allowed": support_passed,
        "future_target_access_allowed": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_safety_fields(),
    }
    manifest["candidate_freeze_manifest_id"] = canonical_json_sha256(manifest)
    _write_json(manifest_path, manifest)
    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "selected_rows_path": str(selected_path),
        "selected_rows_sha256": _sha256_file(selected_path),
        "target_free_support_gate_passed": support_passed,
        "target_free_support_blocking_reason_codes": blockers,
        "selected_side_count": dict(sorted(selected_side_count.items())),
        "strictly_later_outcome_blind_collection_allowed": support_passed,
    }


def build_v6_7_target_free_candidate_rows(
    predictions: list[dict[str, Any]],
    *,
    action_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply every hard pre-exposure execution check except the p_up signal veto."""

    action_by_key = {_action_key(row): row for row in action_rows}
    guard = profile["hard_execution_safety"]
    candidate_rows = []
    excluded = Counter()
    for prediction in predictions:
        action = str(prediction.get("action") or "")
        if action not in SBC_ACTIONS:
            excluded["action_not_sell_before_close"] += 1
            continue
        source = action_by_key.get(_action_key(prediction))
        if source is None:
            raise ValueError("#227 target-free action row identity missing")
        reasons = _microstructure_blocking_reasons(source, guard=guard)
        score = _finite_float(prediction.get("mean_ev_lower_confidence_bound"))
        if score is None:
            reasons.append("mean_ev_lower_confidence_bound_missing_or_non_finite")
        elif score <= float(profile["selection_rule"]["entry_threshold"]):
            reasons.append("mean_ev_lower_confidence_bound_not_positive")
        if reasons:
            excluded.update(set(reasons))
            continue
        row = {
            "schema_version": ROW_SCHEMA_VERSION,
            "market_id": str(source["market_id"]),
            "market_slug": str(source.get("market_slug") or ""),
            "decision_ts": int(source["decision_ts"]),
            "market_close_ts": int(source["market_close_ts"]),
            "max_input_ts": int(source["max_input_ts"]),
            "action": action,
            "side": str(source["side"]),
            "action_family": "SELL_BEFORE_CLOSE",
            "v6_7_selection_score": float(score),
            "v6_7_selection_score_source": (
                "frozen_v6_2_market_clustered_mean_ev_lower_confidence_bound"
            ),
            "calibrated_action_expected_net_return": float(
                prediction["calibrated_action_expected_net_return"]
            ),
            "raw_pairwise_rank_score": float(
                prediction["raw_pairwise_rank_score"]
            ),
            "p_up": float(source["p_up"]),
            "p_down": float(source["p_down"]),
            "selected_side_probability": float(
                source["selected_side_probability"]
            ),
            "p_up_action_disagreement": bool(
                prediction["p_up_action_disagreement"]
            ),
            "p_up_action_disagreement_diagnostic_only": True,
            "p_up_side_alignment_filter_enabled": False,
            "microstructure_snapshot": dict(source["microstructure_snapshot"]),
            "decision_time_features": dict(source["decision_time_features"]),
            "reference_price_feature_provenance": dict(
                source["reference_price_feature_provenance"]
            ),
            "microstructure_safety_passed": True,
            "microstructure_blocking_reason_codes": [],
            "hard_execution_safety_thresholds_unchanged": True,
            "exposure_duplicate_position_and_sizing_guards_unchanged": True,
            "source_score_mutated": False,
            "target_fields_used_for_selection": False,
            "labels_outcomes_resolution_or_pnl_opened": False,
            **_blocked_safety_fields(),
        }
        row["candidate_row_sha256"] = canonical_json_sha256(row)
        candidate_rows.append(row)
    candidate_rows.sort(
        key=lambda row: (
            int(row["decision_ts"]),
            str(row["market_id"]),
            str(row["action"]),
        )
    )
    side_count = Counter(str(row["side"]) for row in candidate_rows)
    return candidate_rows, {
        "eligible_candidate_row_count": len(candidate_rows),
        "eligible_candidate_market_count": len(
            {str(row["market_id"]) for row in candidate_rows}
        ),
        "eligible_candidate_side_row_count": dict(sorted(side_count.items())),
        "excluded_reason_distribution": dict(sorted(excluded.items())),
    }


def select_v6_7_target_free_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select one deterministic positive-LCB candidate per market."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["market_id"])].append(row)
    selected = []
    for market_id in sorted(grouped):
        winner = sorted(
            grouped[market_id],
            key=lambda row: (
                -float(row["v6_7_selection_score"]),
                int(row["decision_ts"]),
                str(row["action"]),
            ),
        )[0]
        selected.append(
            {
                **winner,
                "market_candidate_count": len(grouped[market_id]),
                "market_selection_rank": 1,
                "selection_tie_break": list(profile["selection_rule"]["tie_break"]),
            }
        )
    selected.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
    return selected


def _microstructure_blocking_reasons(
    row: dict[str, Any], *, guard: dict[str, Any]
) -> list[str]:
    reasons = []
    features = dict(row.get("decision_time_features") or {})
    micro = dict(row.get("microstructure_snapshot") or {})
    provenance = dict(row.get("reference_price_feature_provenance") or {})
    execution_price = _finite_float(features.get("execution_price"))
    if execution_price is None or execution_price <= 0.0 or execution_price >= 1.0:
        reasons.append("execution_price_invalid")
    spread = _finite_float(micro.get("spread_bps"))
    if spread is None or spread > float(guard["max_spread_bps"]):
        reasons.append("execution_spread_too_wide")
    staleness = _finite_float(micro.get("book_staleness_ms"))
    if staleness is None or staleness > float(guard["max_book_staleness_ms"]):
        reasons.append("execution_book_stale")
    queue = _finite_float(micro.get("queue_fill_proxy"))
    if queue is None or queue < float(guard["min_queue_fill_probability_proxy"]):
        reasons.append("execution_liquidity_too_weak")
    time_to_close = _finite_float(micro.get("time_to_close_seconds"))
    if time_to_close is None or time_to_close < float(
        guard["min_time_to_close_seconds"]
    ):
        reasons.append("execution_time_to_close_unsafe")
    for field, reason in (
        ("selected_side_executable_ask_notional", "executable_ask_notional_missing"),
        ("selected_side_executable_bid_notional", "executable_bid_notional_missing"),
        ("selected_side_liquidity_depth", "liquidity_depth_missing"),
    ):
        value = _finite_float(features.get(field))
        if value is None or value <= 0.0:
            reasons.append(reason)
    decision_ts = int(row.get("decision_ts") or 0)
    max_input_ts = int(row.get("max_input_ts") or 0)
    if decision_ts <= 0 or max_input_ts <= 0 or max_input_ts > decision_ts:
        reasons.append("feature_timestamp_causality_violation")
    if provenance.get("provenance_valid") is not True:
        reasons.append("reference_provenance_invalid")
    return sorted(set(reasons))


def _legacy_guard_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_trade = [
        row
        for row in rows
        if row.get("selected_action_family") == "SELL_BEFORE_CLOSE"
        and row.get("source_selected_action") in SBC_ACTIONS
    ]
    accepted = [row for row in source_trade if row.get("execution_guard_order_allowed")]
    return {
        "source_selected_sbc_market_count": len(
            {str(row["market_id"]) for row in source_trade}
        ),
        "source_selected_sbc_side_count": dict(
            sorted(Counter(str(row["selected_side"]) for row in source_trade).items())
        ),
        "guard_accepted_sbc_market_count": len(
            {str(row["market_id"]) for row in accepted}
        ),
        "guard_accepted_sbc_side_count": dict(
            sorted(Counter(str(row["selected_side"]) for row in accepted).items())
        ),
        "blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in source_trade
                    for reason in row.get("execution_blocking_reason_codes") or []
                ).items()
            )
        ),
    }


def _validate_source_freeze_manifest(
    manifest: dict[str, Any], *, expected: dict[str, str]
) -> None:
    checks = {
        "labels sealed": manifest.get("labels_outcomes_resolution_or_pnl_opened")
        is False,
        "future target blocked": manifest.get("future_target_access_allowed") is False,
        "predictions": _manifest_sha(manifest, "v6_2_target_free_predictions")
        == expected["predictions"],
        "five action rows": _manifest_sha(manifest, "target_free_five_action_rows")
        == expected["five_action_rows"],
        "legacy replay": _manifest_sha(manifest, "v6_2_outcome_blind_guard_replay")
        == expected["legacy_guard_replay"],
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#227 source freeze invalid: " + ", ".join(blockers))


def _validate_target_free_inputs(
    predictions: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    legacy_replay: list[dict[str, Any]],
) -> None:
    if len(predictions) != len(action_rows):
        raise ValueError("#227 prediction/action row count mismatch")
    if {_action_key(row) for row in predictions} != {
        _action_key(row) for row in action_rows
    }:
        raise ValueError("#227 prediction/action identity mismatch")
    market_count = len({str(row["market_id"]) for row in action_rows})
    if market_count != 60:
        raise ValueError("#227 target-free source is not exact 60 markets")
    for collection in (predictions, action_rows, legacy_replay):
        for row in collection:
            forbidden = FORBIDDEN_TARGET_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(
                    "#227 target-free input contains forbidden fields: "
                    + ", ".join(sorted(forbidden))
                )
            for attestation in (
                "target_or_outcome_fields_used",
                "target_used_as_decision_input",
                "outcome_fields_used_as_decision_input",
            ):
                if row.get(attestation) is True:
                    raise ValueError(f"#227 target-free attestation failed: {attestation}")


def _report_markdown(report: dict[str, Any]) -> str:
    legacy = report["legacy_guard_summary"]
    return "\n".join(
        [
            "# v6.7 p_up Semantic Compatibility",
            "",
            f"- run_id: `{report['run_id']}`",
            "- outcome access: `false`",
            "- score source: `frozen_v6_2_market_clustered_mean_ev_lcb`",
            "- p_up side-alignment hard filter: `false`",
            "- p_up disagreement: `diagnostic_only`",
            "- hard execution safety thresholds changed: `false`",
            f"- legacy selected side count: `{legacy['source_selected_sbc_side_count']}`",
            f"- legacy accepted side count: `{legacy['guard_accepted_sbc_side_count']}`",
            f"- v6.7 selected side count: `{report['v6_7_selected_side_count']}`",
            "- v6.7 selected p_up disagreement count: "
            f"`{report['v6_7_selected_p_up_disagreement_count']}`",
            "- target-free support gate passed: "
            f"`{str(report['target_free_support_gate_passed']).lower()}`",
            "- blockers: "
            f"`{report['target_free_support_blocking_reason_codes']}`",
            "- strictly-later outcome-blind collection allowed: "
            f"`{str(report['strictly_later_outcome_blind_collection_allowed']).lower()}`",
            "- future target access allowed: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _action_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["market_id"]), int(row["decision_ts"]), str(row["action"])


def _manifest_sha(manifest: dict[str, Any], key: str) -> str:
    return str(dict(manifest.get(key) or {}).get("sha256") or "")


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
