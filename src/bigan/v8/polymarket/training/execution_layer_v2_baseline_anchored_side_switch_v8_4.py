"""Baseline-anchored side-switch evidence gate for issue #251."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_abstention_aware_expected_net_pnl_v7_0 import (
    _v7_0_blocked_safety_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    _descriptor,
    _load_json,
    _load_jsonl,
    _require_git_sha,
    _require_sha256,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

CANDIDATE_NAME = "baseline_anchored_side_switch_v8_4"
PROFILE_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-profile-v1"
)
EVIDENCE_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-evidence-v1"
)
DECISION_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-decision-v1"
)
EVIDENCE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-evidence-report-v1"
)
HISTORICAL_REPORT_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-historical-report-v1"
)
MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-baseline-anchored-side-switch-v8-4-manifest-v1"
)
SWITCH_CLASSES = ("AGGREGATE", "UP_TO_DOWN", "DOWN_TO_UP")
TRADE_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
}


@dataclass(frozen=True, slots=True)
class BaselineAnchoredSideSwitchV84Config:
    """Pinned inputs for the historical-only v8.4 evidence freeze."""

    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    v8_3_historical_manifest_path: Path | str
    expected_v8_3_historical_manifest_sha256: str
    issue250_target_free_selected_rows_path: Path | str
    expected_issue250_target_free_selected_rows_sha256: str
    implementation_commit: str
    evidence_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        _require_sha256(self.expected_profile_sha256, "expected_profile_sha256")
        _require_sha256(
            self.expected_v8_3_historical_manifest_sha256,
            "expected_v8_3_historical_manifest_sha256",
        )
        _require_sha256(
            self.expected_issue250_target_free_selected_rows_sha256,
            "expected_issue250_target_free_selected_rows_sha256",
        )
        _require_git_sha(self.implementation_commit)
        if self.evidence_created_ts <= 0:
            raise ValueError("evidence_created_ts must be positive")
        for name in (
            "output_dir",
            "profile_path",
            "v8_3_historical_manifest_path",
            "issue250_target_free_selected_rows_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


def validate_baseline_anchored_side_switch_v8_4_profile(
    profile: dict[str, Any],
) -> None:
    """Reject drift from the issue #251 preregistration."""

    checks = {
        "identity": profile.get("schema_version") == PROFILE_SCHEMA_VERSION
        and profile.get("issue_number") == 251
        and profile.get("candidate_name") == CANDIDATE_NAME
        and profile.get("preregistered_in_issue_before_implementation") is True
        and profile.get("frozen") is True,
        "baseline": profile.get("baseline_contract")
        == {
            "champion": "p_up_semantic_compatibility_v6_7",
            "champion_is_default_on_disagreement": True,
            "accepted_support_may_fall_below_champion": False,
            "source_score_mutation_allowed": False,
            "full_execution_guard_unchanged": True,
        },
        "selection": profile.get("selection_contract")
        == {
            "agreement_behavior": "keep_common_guard_passed_action",
            "eligible_disagreement_behavior": (
                "allow_v8_1_side_switch_after_independent_full_guard"
            ),
            "ineligible_disagreement_behavior": "keep_guard_passed_v6_7",
            "missing_or_invalid_evidence_behavior": "keep_guard_passed_v6_7",
            "baseline_blocked_and_switch_ineligible_behavior": "NO_TRADE",
            "side_quota_enabled": False,
        },
        "evidence": profile.get("switch_evidence_contract")
        == {
            "classes": list(SWITCH_CLASSES),
            "target": (
                "v8_1_switch_after_cost_pnl_minus_v6_7_after_cost_pnl"
            ),
            "bootstrap_unit": "market_id",
            "bootstrap_seed": 2512026,
            "bootstrap_resample_count": 10000,
            "bootstrap_confidence_level": 0.95,
            "point_incremental_pnl_minimum_exclusive": 0.0,
            "bootstrap_lower_bound_minimum_exclusive": 0.0,
            "largest_winner_removed_minimum_inclusive": 0.0,
            "leave_one_market_out_minimum_inclusive": 0.0,
            "aggregate_and_direction_class_must_both_pass": True,
            "threshold_search_enabled": False,
            "result_selected_rerun_allowed": False,
        },
        "historical_gate": profile.get("historical_noninferiority_gate")
        == {
            "candidate_support_not_below_v6_7": True,
            "candidate_minus_v6_7_total_pnl_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_largest_winner_removed_minimum_inclusive": 0.0,
            "candidate_minus_v6_7_market_bootstrap_lcb_minimum_inclusive": 0.0,
            "equality_passes": True,
            "promotion_or_paper_unlock_allowed": False,
        },
        "exclusion": profile.get("future_result_exclusion")
        == {
            "issue250_target_free_selected_rows_sha256": (
                "2330db081fa6ddd13df2c79d999c2bd1c7782c7c91e9d859c943b1c870c3e092"
            ),
            "issue250_terminal_pnl_gate_manifest_sha256": (
                "6ecb044f3fc34c4ebf2063412da71fe974f2f55f231fb8fa51418cf247d5ae26"
            ),
            "issue250_outcomes_or_pnl_used_for_fit": False,
            "issue250_outcomes_or_pnl_used_for_threshold_selection": False,
            "issue249_or_issue246_future_outcomes_used": False,
        },
        "lineage": profile.get("lineage")
        == {
            "v8_3_historical_manifest_sha256": (
                "adb930dc8bde72d89ae8b7520907ad88bb29d54f9c3b22317f9f4635ce5e015d"
            ),
            "v8_3_profile_sha256": (
                "84c6bc06db0c2d25d342ecda23f5c06a4d9809c39db94a8eca1e550a4f822088"
            ),
            "v6_7_profile_sha256": (
                "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
            ),
        },
        "safety": profile.get("safety") == _v7_0_blocked_safety_fields(),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#251 v8.4 profile invalid: " + ", ".join(blockers))


def build_side_switch_evidence_artifact(
    *,
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    profile: dict[str, Any],
    evidence_created_ts: int,
    source_manifest_sha256: str,
    excluded_future_registry_hash: str,
) -> dict[str, Any]:
    """Freeze historical incremental evidence without future outcomes."""

    validate_baseline_anchored_side_switch_v8_4_profile(profile)
    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    _require_sha256(excluded_future_registry_hash, "excluded_future_registry_hash")
    examples = _switch_examples(candidate_rows, baseline_rows)
    contract = profile["switch_evidence_contract"]
    metrics = {
        name: _switch_metrics(
            (
                examples
                if name == "AGGREGATE"
                else [row for row in examples if row["switch_class"] == name]
            ),
            contract=contract,
            seed_offset=index,
        )
        for index, name in enumerate(SWITCH_CLASSES)
    }
    artifact = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "frozen": True,
        "decision_time_safe": True,
        "evidence_created_ts": evidence_created_ts,
        "source_historical_manifest_sha256": source_manifest_sha256.lower(),
        "excluded_future_registry_hash": excluded_future_registry_hash.lower(),
        "historical_targets_used_for_evidence_fit": True,
        "issue250_outcomes_or_pnl_used_for_fit": False,
        "issue250_outcomes_or_pnl_used_for_threshold_selection": False,
        "future_outcomes_used_for_fit_or_selection": False,
        "source_scores_mutated": False,
        "execution_guard_changed": False,
        "switch_example_count": len(examples),
        "switch_class_metrics": metrics,
        "aggregate_switch_eligible": metrics["AGGREGATE"]["eligible"],
        "direction_class_eligibility": {
            name: metrics[name]["eligible"] for name in SWITCH_CLASSES[1:]
        },
        "eligible_switch_classes": [
            name for name in SWITCH_CLASSES if metrics[name]["eligible"]
        ],
    }
    artifact["artifact_id"] = canonical_json_sha256(artifact)
    validate_side_switch_evidence_artifact(artifact, profile=profile)
    return artifact


def validate_side_switch_evidence_artifact(
    artifact: dict[str, Any],
    *,
    profile: dict[str, Any],
) -> None:
    """Validate frozen switch eligibility before inference."""

    validate_baseline_anchored_side_switch_v8_4_profile(profile)
    metrics = artifact.get("switch_class_metrics")
    checks = {
        "identity": artifact.get("schema_version") == EVIDENCE_SCHEMA_VERSION
        and artifact.get("candidate_name") == CANDIDATE_NAME,
        "freeze": artifact.get("frozen") is True
        and artifact.get("decision_time_safe") is True,
        "future_exclusion": artifact.get(
            "issue250_outcomes_or_pnl_used_for_fit"
        )
        is False
        and artifact.get(
            "issue250_outcomes_or_pnl_used_for_threshold_selection"
        )
        is False
        and artifact.get("future_outcomes_used_for_fit_or_selection") is False,
        "mutation": artifact.get("source_scores_mutated") is False
        and artifact.get("execution_guard_changed") is False,
        "classes": isinstance(metrics, dict)
        and set(metrics) == set(SWITCH_CLASSES),
        "artifact_id": artifact.get("artifact_id")
        == canonical_json_sha256(
            {key: value for key, value in artifact.items() if key != "artifact_id"}
        ),
    }
    if isinstance(metrics, dict):
        for name in SWITCH_CLASSES:
            row = metrics.get(name)
            checks[f"metrics_{name}"] = _valid_metric_row(row)
        if set(metrics) == set(SWITCH_CLASSES):
            expected_eligible = [
                name for name in SWITCH_CLASSES if metrics[name]["eligible"]
            ]
            checks["eligibility_consistency"] = (
                artifact.get("aggregate_switch_eligible")
                is metrics["AGGREGATE"]["eligible"]
                and artifact.get("direction_class_eligibility")
                == {
                    name: metrics[name]["eligible"]
                    for name in SWITCH_CLASSES[1:]
                }
                and artifact.get("eligible_switch_classes") == expected_eligible
            )
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ValueError("#251 v8.4 evidence invalid: " + ", ".join(blockers))


def select_baseline_anchored_side_switch_v8_4_decision(
    *,
    candidate_row: dict[str, Any],
    baseline_row: dict[str, Any],
    evidence_artifact: dict[str, Any] | None,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Select a guarded side while preserving v6.7 on unsupported disagreement."""

    validate_baseline_anchored_side_switch_v8_4_profile(profile)
    _assert_inference_target_free(candidate_row, baseline_row)
    candidate_market = str(candidate_row.get("market_id") or "")
    baseline_market = str(baseline_row.get("market_id") or "")
    if not candidate_market or candidate_market != baseline_market:
        raise ValueError("#251 candidate/baseline market identity mismatch")

    candidate_action = _action(candidate_row)
    baseline_action = _action(baseline_row)
    candidate_side = _side(candidate_row)
    baseline_side = _side(baseline_row)
    candidate_allowed = candidate_row.get("execution_guard_order_allowed") is True
    baseline_allowed = baseline_row.get("execution_guard_order_allowed") is True
    evidence_valid = True
    try:
        if evidence_artifact is None:
            raise ValueError("missing evidence")
        validate_side_switch_evidence_artifact(evidence_artifact, profile=profile)
    except (KeyError, TypeError, ValueError):
        evidence_valid = False

    agreement = (
        candidate_action == baseline_action and candidate_side == baseline_side
    )
    direction = _switch_class(baseline_side, candidate_side)
    aggregate_eligible = bool(
        evidence_valid and evidence_artifact.get("aggregate_switch_eligible")
    )
    direction_eligible = bool(
        evidence_valid
        and evidence_artifact.get("direction_class_eligibility", {}).get(direction)
    )
    switch_eligible = aggregate_eligible and direction_eligible

    if agreement and baseline_allowed:
        action = baseline_action
        side = baseline_side
        source = "v6_7_v8_1_agreement"
        reasons = ["common_guard_passed_action_preserved"]
        allowed = True
    elif (
        not agreement
        and switch_eligible
        and candidate_allowed
        and candidate_action in TRADE_ACTIONS
    ):
        action = candidate_action
        side = candidate_side
        source = "eligible_v8_1_side_switch"
        reasons = [
            "aggregate_switch_evidence_eligible",
            f"{direction.lower()}_evidence_eligible",
            "candidate_independent_full_guard_passed",
        ]
        allowed = True
    elif baseline_allowed and baseline_action in TRADE_ACTIONS:
        action = baseline_action
        side = baseline_side
        source = "v6_7_baseline_preserved"
        reasons = [
            (
                "switch_evidence_invalid"
                if not evidence_valid
                else "switch_evidence_ineligible"
            ),
            "baseline_independent_full_guard_passed",
        ]
        allowed = True
    elif agreement and candidate_allowed and candidate_action in TRADE_ACTIONS:
        action = candidate_action
        side = candidate_side
        source = "common_candidate_guard_passed"
        reasons = ["baseline_blocked_common_action_candidate_guard_passed"]
        allowed = True
    else:
        action = "NO_TRADE"
        side = "NONE"
        source = "fail_closed_no_trade"
        reasons = [
            (
                "switch_evidence_invalid"
                if not evidence_valid
                else "switch_evidence_ineligible"
            ),
            "guard_passed_v6_7_baseline_unavailable",
        ]
        allowed = False

    row = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "market_id": candidate_market,
        "decision_ts": _decision_ts(candidate_row, baseline_row),
        "original_v8_1_action": candidate_action,
        "original_v8_1_side": candidate_side,
        "original_v8_1_guard_allowed": candidate_allowed,
        "original_v6_7_action": baseline_action,
        "original_v6_7_side": baseline_side,
        "original_v6_7_guard_allowed": baseline_allowed,
        "selected_action": action,
        "selected_side": side,
        "execution_guard_order_allowed": allowed,
        "selection_source": source,
        "selection_reason_codes": reasons,
        "switch_class": direction,
        "switch_evidence_artifact_valid": evidence_valid,
        "aggregate_switch_evidence_eligible": aggregate_eligible,
        "direction_switch_evidence_eligible": direction_eligible,
        "source_score_mutated": False,
        "full_execution_guard_unchanged": True,
        "target_or_outcome_used_for_selection": False,
        **_v7_0_blocked_safety_fields(),
    }
    row["decision_id"] = canonical_json_sha256(row)
    return row


def build_v8_4_historical_replay(
    *,
    decision_rows: list[dict[str, Any]],
    candidate_target_rows: list[dict[str, Any]],
    baseline_target_rows: list[dict[str, Any]],
    evidence_artifact: dict[str, Any],
    profile: dict[str, Any],
    evaluation_started_ts: int,
) -> dict[str, Any]:
    """Replay the frozen evidence artifact on its historical screening corpus."""

    validate_side_switch_evidence_artifact(evidence_artifact, profile=profile)
    candidate_targets = _target_index(candidate_target_rows)
    baseline_targets = _target_index(baseline_target_rows)
    if set(candidate_targets) != set(baseline_targets):
        raise ValueError("#251 historical candidate/baseline market sets differ")
    decisions_by_market = {
        str(row.get("market_id") or ""): row for row in decision_rows
    }
    if set(decisions_by_market) != set(candidate_targets):
        raise ValueError("#251 historical decision/target market sets differ")

    selected_rows: list[dict[str, Any]] = []
    frozen_decisions: list[dict[str, Any]] = []
    market_order = sorted(
        candidate_targets,
        key=lambda value: (
            int(decisions_by_market[value].get("decision_ts") or 0),
            value,
        ),
    )
    for market_id in market_order:
        source = decisions_by_market[market_id]
        candidate_guard = _candidate_guard_from_overlay(source)
        baseline_guard = _baseline_guard_from_overlay(source)
        decision = select_baseline_anchored_side_switch_v8_4_decision(
            candidate_row=candidate_guard,
            baseline_row=baseline_guard,
            evidence_artifact=evidence_artifact,
            profile=profile,
        )
        frozen_decisions.append(decision)
        target = (
            candidate_targets[market_id]
            if (
                decision["selected_action"]
                == candidate_targets[market_id].get("action")
                and decision["selected_side"]
                == candidate_targets[market_id].get("side")
            )
            else baseline_targets[market_id]
        )
        pnl = (
            float(target["after_cost_net_pnl_at_frozen_size"])
            if decision["execution_guard_order_allowed"]
            else 0.0
        )
        selected_rows.append(
            {
                "market_id": market_id,
                "decision_ts": decision["decision_ts"],
                "action": decision["selected_action"],
                "side": decision["selected_side"],
                "selection_source": decision["selection_source"],
                "after_cost_net_pnl_at_frozen_size": pnl,
                "historical_target_used_for_evidence_fit": True,
                "target_used_as_decision_time_input": False,
            }
        )

    baseline_pnls = [
        float(baseline_targets[market_id]["after_cost_net_pnl_at_frozen_size"])
        for market_id in market_order
    ]
    candidate_pnls = [
        float(row["after_cost_net_pnl_at_frozen_size"])
        for row in selected_rows
    ]
    deltas = [
        candidate_pnls[index] - baseline_pnls[index]
        for index in range(len(candidate_pnls))
    ]
    bootstrap = _bootstrap_interval(
        deltas,
        seed=int(profile["switch_evidence_contract"]["bootstrap_seed"]) + 50,
        resample_count=int(
            profile["switch_evidence_contract"]["bootstrap_resample_count"]
        ),
        confidence_level=float(
            profile["switch_evidence_contract"]["bootstrap_confidence_level"]
        ),
    )
    candidate_total = sum(candidate_pnls)
    baseline_total = sum(baseline_pnls)
    candidate_removed = _largest_winner_removed(candidate_pnls)
    baseline_removed = _largest_winner_removed(baseline_pnls)
    support = sum(
        row["execution_guard_order_allowed"] is True for row in frozen_decisions
    )
    baseline_support = sum(
        row.get("original_v6_7_guard_allowed") is True for row in frozen_decisions
    )
    final_divergences = sum(
        (
            row["selected_action"] != row["original_v6_7_action"]
            or row["selected_side"] != row["original_v6_7_side"]
        )
        for row in frozen_decisions
    )
    checks = {
        "candidate_support_not_below_v6_7": support >= baseline_support,
        "candidate_total_pnl_noninferior_to_v6_7": (
            candidate_total - baseline_total >= 0.0
        ),
        "candidate_largest_winner_removed_noninferior_to_v6_7": (
            candidate_removed - baseline_removed >= 0.0
        ),
        "candidate_minus_v6_7_market_bootstrap_lcb_nonnegative": (
            bootstrap["lower_confidence_bound"] >= 0.0
        ),
        "source_scores_unchanged": all(
            row["source_score_mutated"] is False for row in frozen_decisions
        ),
        "full_execution_guard_unchanged": all(
            row["full_execution_guard_unchanged"] is True
            for row in frozen_decisions
        ),
        "issue250_outcomes_excluded": (
            evidence_artifact["issue250_outcomes_or_pnl_used_for_fit"] is False
        ),
    }
    report = {
        "schema_version": HISTORICAL_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": "p_up_semantic_compatibility_v6_7",
        "evaluation_started_ts": evaluation_started_ts,
        "historical_market_count": len(selected_rows),
        "candidate_guard_accepted_market_count": support,
        "v6_7_guard_accepted_market_count": baseline_support,
        "final_policy_difference_market_count": final_divergences,
        "selection_source_distribution": dict(
            sorted(Counter(row["selection_source"] for row in frozen_decisions).items())
        ),
        "candidate_total_after_cost_pnl": candidate_total,
        "v6_7_total_after_cost_pnl": baseline_total,
        "candidate_minus_v6_7_total_after_cost_pnl": (
            candidate_total - baseline_total
        ),
        "candidate_largest_winner_removed_after_cost_pnl": candidate_removed,
        "v6_7_largest_winner_removed_after_cost_pnl": baseline_removed,
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl": (
            candidate_removed - baseline_removed
        ),
        "candidate_minus_v6_7_market_bootstrap": bootstrap,
        "checks": checks,
        "historical_noninferiority_gate_passed": all(checks.values()),
        "model_improvement_demonstrated": (
            final_divergences > 0 and candidate_total > baseline_total
        ),
        "candidate_decisions_equivalent_to_v6_7": final_divergences == 0,
        "new_future_challenger_collection_justified": final_divergences > 0
        and all(checks.values()),
        "historical_targets_used_for_precollection_screening_only": True,
        "issue250_outcomes_or_pnl_used_for_fit_or_tuning": False,
        "target_used_as_decision_time_input": False,
        **_v7_0_blocked_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return {
        "evidence_artifact": evidence_artifact,
        "frozen_decisions": frozen_decisions,
        "selected_rows": selected_rows,
        "report": report,
    }


def run_baseline_anchored_side_switch_v8_4(
    config: BaselineAnchoredSideSwitchV84Config,
) -> dict[str, Any]:
    """Freeze evidence, replay historically, and write hash-addressed artifacts."""

    paths = {
        "profile": config.profile_path.resolve(),
        "historical": config.v8_3_historical_manifest_path.resolve(),
        "excluded": config.issue250_target_free_selected_rows_path.resolve(),
    }
    pins = {
        "profile": config.expected_profile_sha256,
        "historical": config.expected_v8_3_historical_manifest_sha256,
        "excluded": config.expected_issue250_target_free_selected_rows_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#251 {name}")
    profile = _load_json(paths["profile"])
    validate_baseline_anchored_side_switch_v8_4_profile(profile)
    if pins["historical"].lower() != profile["lineage"][
        "v8_3_historical_manifest_sha256"
    ]:
        raise ValueError("#251 historical lineage pin mismatch")
    if pins["excluded"].lower() != profile["future_result_exclusion"][
        "issue250_target_free_selected_rows_sha256"
    ]:
        raise ValueError("#251 future exclusion pin mismatch")

    historical = _load_json(paths["historical"])
    candidate_targets = _load_jsonl(
        Path(_verified_descriptor(historical["candidate_selected_rows"])["path"])
    )
    baseline_targets = _load_jsonl(
        Path(_verified_descriptor(historical["v6_7_baseline_selected_rows"])["path"])
    )
    decisions = _load_jsonl(
        Path(_verified_descriptor(historical["decision_rows"])["path"])
    )
    excluded_rows = _load_jsonl(paths["excluded"])
    excluded_ids = {
        str(row.get("market_id") or "") for row in excluded_rows
    } - {""}
    historical_ids = {
        str(row.get("market_id") or "") for row in baseline_targets
    } - {""}
    if excluded_ids & historical_ids:
        raise ValueError("#251 historical rows overlap excluded issue250 markets")
    excluded_registry_hash = canonical_json_sha256(sorted(excluded_ids))

    evidence = build_side_switch_evidence_artifact(
        candidate_rows=candidate_targets,
        baseline_rows=baseline_targets,
        profile=profile,
        evidence_created_ts=config.evidence_created_ts,
        source_manifest_sha256=pins["historical"],
        excluded_future_registry_hash=excluded_registry_hash,
    )
    result = build_v8_4_historical_replay(
        decision_rows=decisions,
        candidate_target_rows=candidate_targets,
        baseline_target_rows=baseline_targets,
        evidence_artifact=evidence,
        profile=profile,
        evaluation_started_ts=config.evidence_created_ts,
    )
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists() and not config.overwrite_existing:
        raise FileExistsError(f"run directory exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return _write_outputs(
        run_dir=run_dir,
        config=config,
        profile_path=paths["profile"],
        historical_path=paths["historical"],
        excluded_path=paths["excluded"],
        excluded_registry_hash=excluded_registry_hash,
        result=result,
    )


def _switch_examples(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = _target_index(candidate_rows)
    baseline = _target_index(baseline_rows)
    if set(candidate) != set(baseline):
        raise ValueError("#251 candidate/baseline target market sets differ")
    examples = []
    for market_id in sorted(
        candidate,
        key=lambda value: (
            int(candidate[value].get("decision_ts") or 0),
            value,
        ),
    ):
        candidate_row = candidate[market_id]
        baseline_row = baseline[market_id]
        if (
            candidate_row.get("action") == baseline_row.get("action")
            and candidate_row.get("side") == baseline_row.get("side")
        ):
            continue
        candidate_pnl = _target_pnl(candidate_row)
        baseline_pnl = _target_pnl(baseline_row)
        examples.append(
            {
                "market_id": market_id,
                "decision_ts": int(candidate_row.get("decision_ts") or 0),
                "switch_class": _switch_class(
                    str(baseline_row.get("side") or ""),
                    str(candidate_row.get("side") or ""),
                ),
                "candidate_pnl": candidate_pnl,
                "baseline_pnl": baseline_pnl,
                "incremental_pnl": candidate_pnl - baseline_pnl,
            }
        )
    return examples


def _switch_metrics(
    rows: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    deltas = [float(row["incremental_pnl"]) for row in rows]
    total = sum(deltas)
    bootstrap = _bootstrap_interval(
        deltas,
        seed=int(contract["bootstrap_seed"]) + seed_offset,
        resample_count=int(contract["bootstrap_resample_count"]),
        confidence_level=float(contract["bootstrap_confidence_level"]),
    )
    largest = max(deltas) if deltas else None
    removed = total - largest if largest is not None else 0.0
    leave_one_out = (
        min(total - value for value in deltas) if len(deltas) >= 2 else None
    )
    checks = {
        "nonempty_support": bool(deltas),
        "point_incremental_pnl_positive": total
        > float(contract["point_incremental_pnl_minimum_exclusive"]),
        "bootstrap_lower_bound_positive": bootstrap["lower_confidence_bound"]
        > float(contract["bootstrap_lower_bound_minimum_exclusive"]),
        "largest_winner_removed_nonnegative": removed
        >= float(contract["largest_winner_removed_minimum_inclusive"]),
        "leave_one_market_out_minimum_nonnegative": (
            leave_one_out is not None
            and leave_one_out
            >= float(contract["leave_one_market_out_minimum_inclusive"])
        ),
        "finite_complete_lineage": all(math.isfinite(value) for value in deltas),
    }
    ordered = sorted(rows, key=lambda row: (row["decision_ts"], row["market_id"]))
    midpoint = len(ordered) // 2
    return {
        "unique_market_count": len(rows),
        "incremental_pnl_total": total,
        "incremental_pnl_mean": total / len(rows) if rows else 0.0,
        "positive_delta_count": sum(value > 0.0 for value in deltas),
        "zero_delta_count": sum(value == 0.0 for value in deltas),
        "negative_delta_count": sum(value < 0.0 for value in deltas),
        "positive_sign_rate": (
            sum(value > 0.0 for value in deltas) / len(deltas)
            if deltas
            else 0.0
        ),
        "largest_positive_delta": largest,
        "largest_winner_removed_incremental_pnl": removed,
        "leave_one_market_out_minimum_incremental_pnl": leave_one_out,
        "chronological_first_half_count": midpoint,
        "chronological_second_half_count": len(ordered) - midpoint,
        "market_bootstrap": bootstrap,
        "eligibility_checks": checks,
        "eligible": all(checks.values()),
    }


def _bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    resample_count: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not values:
        return {
            "bootstrap_seed": seed,
            "bootstrap_resample_count": resample_count,
            "confidence_level": confidence_level,
            "point_estimate": 0.0,
            "lower_confidence_bound": 0.0,
            "upper_confidence_bound": 0.0,
        }
    rng = random.Random(seed)
    samples = sorted(
        sum(values[rng.randrange(len(values))] for _ in values)
        for _ in range(resample_count)
    )
    alpha = (1.0 - confidence_level) / 2.0
    low_index = int(alpha * (resample_count - 1))
    high_index = int((1.0 - alpha) * (resample_count - 1))
    return {
        "bootstrap_seed": seed,
        "bootstrap_resample_count": resample_count,
        "confidence_level": confidence_level,
        "point_estimate": sum(values),
        "lower_confidence_bound": samples[low_index],
        "upper_confidence_bound": samples[high_index],
    }


def _valid_metric_row(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    required = {
        "unique_market_count",
        "incremental_pnl_total",
        "market_bootstrap",
        "eligibility_checks",
        "eligible",
    }
    if not required.issubset(row):
        return False
    numeric = [
        row.get("incremental_pnl_total"),
        row.get("incremental_pnl_mean"),
        row.get("largest_winner_removed_incremental_pnl"),
        row.get("positive_sign_rate"),
    ]
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric)


def _target_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ValueError("#251 target market identity missing or duplicated")
        if row.get("target_used_as_decision_time_input") is not False:
            raise ValueError("#251 historical target leakage marker invalid")
        _target_pnl(row)
        output[market_id] = row
    return output


def _target_pnl(row: dict[str, Any]) -> float:
    value = float(row["after_cost_net_pnl_at_frozen_size"])
    if not math.isfinite(value):
        raise ValueError("#251 historical target PnL must be finite")
    return value


def _switch_class(baseline_side: str, candidate_side: str) -> str:
    key = (baseline_side, candidate_side)
    if key == ("UP", "DOWN"):
        return "UP_TO_DOWN"
    if key == ("DOWN", "UP"):
        return "DOWN_TO_UP"
    if baseline_side == candidate_side and baseline_side in {"UP", "DOWN"}:
        return "AGREEMENT"
    return "INVALID"


def _action(row: dict[str, Any]) -> str:
    return str(row.get("selected_action") or row.get("action") or "NO_TRADE")


def _side(row: dict[str, Any]) -> str:
    return str(row.get("selected_side") or row.get("side") or "NONE")


def _decision_ts(*rows: dict[str, Any]) -> int:
    values = [int(row.get("decision_ts") or 0) for row in rows]
    return max(values)


def _assert_inference_target_free(*rows: dict[str, Any]) -> None:
    forbidden = _find_nonempty_fields(rows, FORBIDDEN_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(
            "#251 inference rows contain target fields: " + ",".join(forbidden)
        )


def _find_nonempty_fields(
    value: Any,
    forbidden: set[str],
    *,
    prefix: str = "",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden and item not in (None, "", [], {}):
                found.append(path)
            found.extend(_find_nonempty_fields(item, forbidden, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(
                _find_nonempty_fields(item, forbidden, prefix=f"{prefix}[{index}]")
            )
    return sorted(found)


def _candidate_guard_from_overlay(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row.get("original_v8_1_action"),
        "selected_side": row.get("original_v8_1_side"),
        "execution_guard_order_allowed": row.get(
            "original_v8_1_guard_allowed"
        ),
    }


def _baseline_guard_from_overlay(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_id": row["market_id"],
        "decision_ts": row.get("decision_ts"),
        "selected_action": row.get("original_v6_7_action"),
        "selected_side": row.get("original_v6_7_side"),
        "execution_guard_order_allowed": row.get(
            "original_v6_7_guard_allowed"
        ),
    }


def _largest_winner_removed(values: list[float]) -> float:
    return sum(values) - max(values) if values else 0.0


def _verified_descriptor(value: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(value.get("path") or ""))
    sha256 = str(value.get("sha256") or "")
    _require_sha256(sha256, "descriptor sha256")
    _verify_pin(path, sha256, "#251 descriptor")
    return {"path": str(path.resolve()), "sha256": sha256.lower()}


def _write_outputs(
    *,
    run_dir: Path,
    config: BaselineAnchoredSideSwitchV84Config,
    profile_path: Path,
    historical_path: Path,
    excluded_path: Path,
    excluded_registry_hash: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    evidence = result["evidence_artifact"]
    historical_report = result["report"]
    evidence_path = run_dir / "v8_4_side_switch_evidence_artifact.json"
    evidence_report_path = run_dir / "v8_4_side_switch_evidence_report.json"
    evidence_md_path = run_dir / "v8_4_side_switch_evidence_report.md"
    decisions_path = run_dir / "v8_4_historical_frozen_decisions.jsonl"
    selected_path = run_dir / "v8_4_historical_selected_rows.jsonl"
    historical_report_path = run_dir / "v8_4_historical_noninferiority_report.json"
    historical_md_path = run_dir / "v8_4_historical_noninferiority_report.md"

    evidence_report = {
        "schema_version": EVIDENCE_REPORT_SCHEMA_VERSION,
        "candidate_name": CANDIDATE_NAME,
        "switch_example_count": evidence["switch_example_count"],
        "switch_class_metrics": evidence["switch_class_metrics"],
        "eligible_switch_classes": evidence["eligible_switch_classes"],
        "aggregate_switch_eligible": evidence["aggregate_switch_eligible"],
        "direction_class_eligibility": evidence["direction_class_eligibility"],
        "issue250_outcomes_or_pnl_used_for_fit_or_tuning": False,
        "excluded_future_registry_hash": excluded_registry_hash,
        "new_future_collection_allowed": historical_report[
            "new_future_challenger_collection_justified"
        ],
        **_v7_0_blocked_safety_fields(),
    }
    evidence_report["report_id"] = canonical_json_sha256(evidence_report)
    _write_json(evidence_path, evidence)
    _write_json(evidence_report_path, evidence_report)
    _write_jsonl(decisions_path, result["frozen_decisions"])
    _write_jsonl(selected_path, result["selected_rows"])
    _write_json(historical_report_path, historical_report)
    _write_text(
        evidence_md_path,
        "\n".join(
            [
                "# v8.4 Side-Switch Evidence",
                "",
                f"- switch examples: `{evidence['switch_example_count']}`",
                f"- aggregate eligible: `{str(evidence['aggregate_switch_eligible']).lower()}`",
                f"- direction eligibility: `{evidence['direction_class_eligibility']}`",
                f"- eligible classes: `{evidence['eligible_switch_classes']}`",
                "- #250 outcomes/PnL used for fit or tuning: `false`",
                "",
            ]
        ),
    )
    _write_text(
        historical_md_path,
        "\n".join(
            [
                "# v8.4 Historical Non-Inferiority",
                "",
                f"- markets: `{historical_report['historical_market_count']}`",
                "- candidate minus v6.7 PnL: "
                f"`{historical_report['candidate_minus_v6_7_total_after_cost_pnl']}`",
                "- final policy differences: "
                f"`{historical_report['final_policy_difference_market_count']}`",
                "- historical gate passed: "
                f"`{str(historical_report['historical_noninferiority_gate_passed']).lower()}`",
                "- model improvement demonstrated: "
                f"`{str(historical_report['model_improvement_demonstrated']).lower()}`",
                "- new future challenger collection justified: "
                f"`{str(historical_report['new_future_challenger_collection_justified']).lower()}`",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(profile_path),
        "v8_3_historical_manifest": _descriptor(historical_path),
        "issue250_target_free_exclusion_rows": _descriptor(excluded_path),
        "issue250_excluded_future_registry_hash": excluded_registry_hash,
        "side_switch_evidence_artifact": _descriptor(evidence_path),
        "side_switch_evidence_report": _descriptor(evidence_report_path),
        "side_switch_evidence_report_markdown": _descriptor(evidence_md_path),
        "historical_frozen_decisions": _descriptor(decisions_path),
        "historical_selected_rows": _descriptor(selected_path),
        "historical_noninferiority_report": _descriptor(historical_report_path),
        "historical_noninferiority_report_markdown": _descriptor(
            historical_md_path
        ),
        "historical_noninferiority_gate_passed": historical_report[
            "historical_noninferiority_gate_passed"
        ],
        "model_improvement_demonstrated": historical_report[
            "model_improvement_demonstrated"
        ],
        "new_future_challenger_collection_justified": historical_report[
            "new_future_challenger_collection_justified"
        ],
        "issue250_outcomes_or_pnl_used_for_fit_or_tuning": False,
        "source_scores_mutated": False,
        "execution_guard_changed": False,
        **_v7_0_blocked_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v8_4_historical_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "evidence_artifact_path": evidence_path,
        "evidence_artifact_sha256": _descriptor(evidence_path)["sha256"],
        "evidence_report_path": evidence_report_path,
        "evidence_report_sha256": _descriptor(evidence_report_path)["sha256"],
        "historical_report_path": historical_report_path,
        "historical_report_sha256": _descriptor(historical_report_path)["sha256"],
        "manifest_path": manifest_path,
        "manifest_sha256": _descriptor(manifest_path)["sha256"],
        "evidence_artifact": evidence,
        "historical_report": historical_report,
        "manifest": manifest,
    }
