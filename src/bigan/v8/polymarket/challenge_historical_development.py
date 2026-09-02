"""Governed outcome-aware development for the v8.1 challenge model."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ATTEMPT_CLOSURE_SCHEMA_VERSION = "bigan-v8-challenge-attempt-closure-v1"
DEVELOPMENT_REGISTRY_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-data-registry-v1"
)
SUCCESS_STANDARD_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-success-standard-v1"
)
SUCCESS_STANDARD_V2_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-success-standard-v2"
)
SCALE_INVARIANCE_GOVERNANCE_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-scale-invariance-governance-v1"
)
ATTEMPT_002_SUPERSESSION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-execution-manifest-supersession-v1"
)
ITERATION_LEDGER_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-iteration-ledger-v1"
)
ITERATION_PREREGISTRATION_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-preregistration-v1"
)
ITERATION_RESULT_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-result-v1"
)
ITERATION_ENTRY_SCHEMA_VERSION = (
    "bigan-v8-challenge-historical-development-iteration-entry-v1"
)

ZERO_SHA256 = "0" * 64
FROZEN_COLLECTION_PLAN_SHA256 = (
    "df9a81b1ed8cc9e3bd50ca580cd617ecde041a97a7d8470d0c5b82e9d79436ff"
)
FROZEN_SUPERSESSION_GOVERNANCE_SHA256 = (
    "b6d8eab500a1ed070827b3dceef0101aa3139db1aa6765d2648d4f9c07294d04"
)
LEGACY_EVIDENCE_ENTRY_SHA256S = {
    "36d7aac02912366f3c6c143e608f4eab6e3912748a85b03ecc247cf3c9463ac9",
    "2f041c330677a71aaa43d67c473ae48973c0340918e5f404449c824e60d86056",
    "405d02b091967c71f3c7405e561af190ff37dd70ca11ee995d0a435ea179056f",
    "d2bbdfa67d994e2ea27fe3fb3d06e1c10b36605c4109f4dc865f78971fcec7bc",
    "919ecc8dd179c840630cb1424320ac46d31f5f880c4caf52ac70962849352c20",
}
OUTCOME_AWARE_EVALUATION_IDS = {
    "issue-238-retained-v6-7-future",
    "issue-241-v7-7-future",
    "issue-246-v8-1-future",
    "issue-249-v8-3-future",
    "issue-250-hardened-future",
    "issue-260-exact-120-historical-replay",
    "issue-252-v6-7-exact-195-replay",
    "issue-260-v8-1-exact-195-diagnostic",
}
SAFE_FALSES = {
    "capital_at_risk": False,
    "collection_start_allowed": False,
    "handoff_allowed": False,
    "live_allowed": False,
    "paper_allowed": False,
    "promotion_allowed": False,
    "wallet_allowed": False,
    "write_allowed": False,
}


class ChallengeHistoricalDevelopmentError(ValueError):
    """Raised when governed historical-development evidence is invalid."""


@dataclass(frozen=True)
class HistoricalDevelopmentEvaluationConfig:
    """Pinned inputs for one preregistered historical-development evaluation."""

    run_id: str
    output_dir: Path
    iteration_number: int
    candidate_id: str
    comparison_rows_path: Path
    expected_comparison_rows_sha256: str
    preregistration_path: Path
    expected_preregistration_sha256: str
    success_standard_path: Path
    expected_success_standard_sha256: str
    registry_path: Path
    expected_registry_sha256: str
    ledger_root_path: Path
    expected_ledger_root_sha256: str
    attempt_closure_path: Path
    expected_attempt_closure_sha256: str
    implementation_base_commit: str
    preregistration_commit: str
    implementation_commit: str
    evaluated_at: str
    previous_iteration_entry_sha256: str = ZERO_SHA256
    previous_iteration_entry_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.candidate_id:
            raise ValueError("run_id and candidate_id are required")
        for commit in (
            self.implementation_base_commit,
            self.preregistration_commit,
            self.implementation_commit,
        ):
            if not _is_git_commit(commit):
                raise ValueError("base, preregistration, and implementation commits are required")
        if not 1 <= self.iteration_number <= 5:
            raise ValueError("iteration_number must be between 1 and 5")
        for digest in (
            self.expected_comparison_rows_sha256,
            self.expected_preregistration_sha256,
            self.expected_success_standard_sha256,
            self.expected_registry_sha256,
            self.expected_ledger_root_sha256,
            self.expected_attempt_closure_sha256,
            self.previous_iteration_entry_sha256,
        ):
            _require_sha256(digest, field="evaluation config digest")
        if self.iteration_number == 1:
            if (
                self.previous_iteration_entry_sha256 != ZERO_SHA256
                or self.previous_iteration_entry_path is not None
            ):
                raise ValueError("iteration 1 must start from the zero-hash ledger genesis")
        elif (
            self.previous_iteration_entry_sha256 == ZERO_SHA256
            or self.previous_iteration_entry_path is None
        ):
            raise ValueError("iterations 2-5 require the previous iteration entry")


def validate_attempt_001_closure(
    closure: Mapping[str, Any],
    *,
    expected_collection_plan_sha256: str,
    expected_supersession_governance_sha256: str,
) -> None:
    """Validate the additive attempt-001 closure without mutating frozen artifacts."""

    lineage = dict(closure.get("frozen_lineage") or {})
    reason = dict(closure.get("closure_reason") or {})
    raw = dict(closure.get("raw_collection_audit") or {})
    disposition = dict(
        (closure.get("alpha_spending_review") or {}).get("attempt_001_disposition")
        or {}
    )
    evaluations = list(
        (closure.get("alpha_spending_review") or {}).get(
            "outcome_aware_evaluations"
        )
        or []
    )
    expected_rate = 23 / 195
    expected_future = 120 * expected_rate
    checks = {
        "schema": closure.get("schema_version") == ATTEMPT_CLOSURE_SCHEMA_VERSION,
        "attempt_id": closure.get("attempt_id")
        == (
            "v8-5-challenger-parallel-future-attempt-001-"
            "provider-complete-corrective-refreeze"
        ),
        "closure_issue": (closure.get("closure_issue") or {}).get("number") == 262,
        "plan_lineage": lineage.get("collection_plan_sha256")
        == expected_collection_plan_sha256,
        "plan_is_frozen_seq_5": expected_collection_plan_sha256
        == FROZEN_COLLECTION_PLAN_SHA256,
        "governance_lineage": lineage.get("supersession_governance_sha256")
        == expected_supersession_governance_sha256,
        "governance_is_frozen_5_of_5": expected_supersession_governance_sha256
        == FROZEN_SUPERSESSION_GOVERNANCE_SHA256,
        "supersessions_exhausted": lineage.get("supersessions_consumed") == 5,
        "acceptance_count": reason.get("exact_195_accepted_market_count") == 23,
        "market_count": reason.get("exact_195_market_count") == 195,
        "acceptance_rate": _float_equal(
            reason.get("exact_195_acceptance_rate"), expected_rate
        ),
        "expected_future_support": _float_equal(
            reason.get("expected_accepted_markets_in_120"), expected_future
        ),
        "minimum_support": reason.get("frozen_minimum_support") == 40,
        "support_incompatible": reason.get("support_is_mathematically_compatible")
        is False
        and reason.get("minimum_acceptance_rate_required") == 40 / 120
        and _float_equal(
            reason.get("projected_support_shortfall"),
            40 - expected_future,
        )
        and reason.get("reason_code")
        == (
            "frozen_support_incompatible_with_observed_outcome_blind_"
            "acceptance_behavior"
        )
        and expected_future < 40,
        "raw_capture_count": raw.get("raw_indexed_capture_count") == 12,
        "valid_capture_count": raw.get("quality_valid_capture_count") == 0,
        "promotion_capture_count": raw.get("promotion_eligible_capture_count") == 0,
        "outcome_blind": raw.get(
            "labels_outcomes_settlement_or_pnl_opened"
        )
        is False,
        "raw_excluded": raw.get("all_raw_captures_permanently_excluded") is True,
        "raw_lineage": raw.get("collector_index_sha256")
        == "d4ec55d2b8aab5dcc05e66113515bda4b7d3070358b759fe8dbdf2d40f3b68a0"
        and raw.get("batch_canary_report_sha256")
        == "0442a3d5fcbf0e1f7d3613c4064d31fe967cbca5fb4a0a3c3cbc70f3ba9f0840"
        and raw.get("terminal_reason") == "batch_has_zero_quality_valid_markets",
        "target_access_never_started": disposition.get("target_access_claim_written")
        is False
        and disposition.get("promotion_evidence_collection_started") is False
        and disposition.get("attempt_or_alpha_consumed_by_target_access") is False,
        "raw_collection_disclosed": disposition.get(
            "raw_outcome_blind_collection_process_started"
        )
        is True,
        "planned_alpha_disclosed": disposition.get("planned_familywise_window_alpha")
        == 0.025
        and disposition.get("planned_per_candidate_alpha") == 0.0125
        and disposition.get("administratively_closed") is True,
        "outcome_aware_review_complete": {
            item.get("evaluation_id") for item in evaluations
        }
        == OUTCOME_AWARE_EVALUATION_IDS
        and all(item.get("outcomes_opened") is True for item in evaluations),
        "outcome_aware_review_terminal": all(
            item.get("terminal") is True
            and item.get("promotion_evidence_reusable") is False
            for item in evaluations
        ),
        "safety": closure.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("attempt-001 closure", checks)


def validate_historical_development_registry(
    registry: Mapping[str, Any],
    *,
    expected_attempt_closure_sha256: str,
    expected_evidence_ledger_sha256: str,
) -> None:
    """Validate the closed-world registry of already outcome-opened corpora."""

    policy = dict(registry.get("development_policy") or {})
    corpora = list(registry.get("registered_corpora") or [])
    by_id = {str(item.get("corpus_id")): item for item in corpora}
    exact_195 = dict(by_id.get("issue-252-exact-195-outcome-opened-corpus") or {})
    artifacts = dict(exact_195.get("artifacts") or {})
    exact_195_evaluations = {
        str(item.get("candidate_id")): item
        for item in exact_195.get("registered_evaluations") or []
    }
    exact_120 = dict(by_id.get("issue-260-exact-120-historical-screen") or {})
    exact_120_artifacts = dict(exact_120.get("artifacts") or {})
    legacy = [item for item in corpora if str(item.get("corpus_id")).startswith("legacy-")]
    checks = {
        "schema": registry.get("schema_version") == DEVELOPMENT_REGISTRY_SCHEMA_VERSION,
        "issue": registry.get("issue") == 262,
        "closure_lineage": registry.get("attempt_001_closure_sha256")
        == expected_attempt_closure_sha256,
        "closed_world": registry.get("registry_closed_world_at_freeze") is True,
        "frozen_before_replay": registry.get(
            "frozen_before_new_development_replay"
        )
        is True,
        "corpus_count": len(corpora) == 7 and len(by_id) == 7,
        "legacy_count": len(legacy) == 5,
        "legacy_lineage": all(
            item.get("evidence_ledger_sha256") == expected_evidence_ledger_sha256
            and item.get("outcomes_opened") is True
            and item.get("development_only_forever") is True
            and item.get("promotion_evidence_eligible") is False
            for item in legacy
        ),
        "legacy_entries_complete": {
            item.get("evidence_ledger_entry_sha256") for item in legacy
        }
        == LEGACY_EVIDENCE_ENTRY_SHA256S,
        "exact_195_count": exact_195.get("exact_market_count") == 195,
        "exact_195_chronological_sequence": exact_195.get(
            "chronological_market_id_sequence_sha256"
        )
        == "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
        and exact_195.get("chronological_market_id_sequence_sha256_method")
        == "sha256_of_utf8_market_ids_one_per_line_with_final_newline",
        "exact_195_settled_index": (
            artifacts.get("settled_corpus_index") or {}
        ).get("sha256")
        == "2512d19f402aa74522ed7290b9f80701a72ab95e1b9da9837a3b4a9a3b920dcf",
        "exact_195_market_ids": (
            artifacts.get("chronological_market_ids") or {}
        ).get("sha256")
        == "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
        and (artifacts.get("chronological_market_ids") or {}).get("size_bytes")
        == 13065,
        "exact_195_rows": (artifacts.get("five_action_rows") or {}).get("sha256")
        == "134425d9f38ffdebbf72043a8b802e95bbb87eebbe7e8d39bfa5d6f8b98828f7",
        "exact_195_development_only": exact_195.get("outcomes_opened") is True
        and exact_195.get("development_only_forever") is True
        and exact_195.get("promotion_evidence_eligible") is False,
        "exact_195_evaluations": (
            exact_195_evaluations.get("matched_frozen_v6_7") or {}
        ).get("report_sha256")
        == "500b51146f7d9fc0c7284702ba954fd4aac2d36644e8b18f1f61121deb1b0ff4"
        and (
            exact_195_evaluations.get("v8_1_primary_no_fallback") or {}
        ).get("report_sha256")
        == "8d5458ee09a3a4fa28a94042db947caaf8dbe3c0874cf96e0435d34bf20d179a"
        and (
            exact_195_evaluations.get("v8_1_primary_no_fallback") or {}
        ).get("market_comparison_sha256")
        == "fce95987a10b160d7a7e6cdfd3842cc3e3b34dd138eb27093fb9f86b0a790eae",
        "exact_120_registered": exact_120.get("exact_market_count") == 120
        and exact_120.get("outcomes_opened") is True
        and exact_120.get("development_only_forever") is True
        and exact_120.get("promotion_evidence_eligible") is False,
        "exact_120_artifacts": exact_120_artifacts
        == {
            "baseline_rows_sha256": (
                "82acc593d57c37c77da866f989487273865c3501ab39aa8fe346aa15cedb74df"
            ),
            "candidate_rows_sha256": (
                "c26c9164cada8abedcff94c05553cb43bdc9eaa138dbff1293abe09af1f4f59f"
            ),
            "report_sha256": (
                "558a1802513e70426d9bd4c589b2e7736ea0ed11a704f51fdaa613e18ef45988"
            ),
        },
        "historical_never_promotes": policy.get(
            "registered_outcome_opened_corpora_are_development_data_forever"
        )
        is True
        and policy.get("historical_results_can_unlock_promotion") is False
        and policy.get("promotion_evidence_source")
        == "not_yet_collected_future_attempt_002_window_only",
        "iteration_allowed": policy.get("outcome_aware_candidate_iteration_allowed")
        is True
        and tuple(policy.get("permitted_iteration_targets") or ())
        == ("controller", "threshold", "feature", "sizing"),
        "safety": registry.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("historical development registry", checks)


def validate_scale_invariance_governance(
    governance: Mapping[str, Any],
) -> None:
    """Validate the additive record that invalidates iteration 3's old pass."""

    affected = dict(governance.get("affected_historical_result") or {})
    audit = dict(governance.get("audit") or {})
    alpha = dict(governance.get("alpha_spending_ledger_update") or {})
    paired = dict(governance.get("paired_gate_diagnostic") or {})
    conclusion = dict(governance.get("conclusion") or {})
    metrics = dict(governance.get("candidate_only_metric_comparison") or {})
    expected_ratios = {
        "absolute_bootstrap_lcb",
        "first_half_bootstrap_ucb",
        "first_half_total_after_cost_pnl",
        "largest_winner_after_cost_pnl",
        "largest_winner_removed_after_cost_pnl",
        "second_half_bootstrap_ucb",
        "second_half_total_after_cost_pnl",
        "total_after_cost_pnl",
    }
    checks = {
        "schema": governance.get("schema_version")
        == SCALE_INVARIANCE_GOVERNANCE_SCHEMA_VERSION,
        "issue": governance.get("issue") == 262,
        "affected_result": affected.get(
            "iteration_003_result_sha256"
        )
        == "997cd1ad280cdb6ad1125d9c2629d283f47e2765317f04557f4fdfc7eb972790"
        and affected.get("historical_success_claim_valid_after_review") is False
        and affected.get("recorded_iteration_artifacts_rewritten") is False
        and affected.get("superseded_success_standard_sha256")
        == "07609f09692723dd1e650080cfdd29466a7ee8a0f8c30d8378045a8ee3523114",
        "same_trade_set": audit.get("selected_market_action_pairs_identical")
        is True
        and audit.get("selected_market_count_iteration_001") == 5
        and audit.get("selected_market_count_iteration_003") == 5
        and audit.get("baseline_after_cost_pnl_identical_row_by_row") is True,
        "candidate_scale": audit.get("candidate_position_size_iteration_001")
        == 0.2
        and audit.get("candidate_position_size_iteration_003") == 1.0
        and audit.get("candidate_only_metric_scale_factor") == 5.0
        and set(metrics) == expected_ratios
        and all(
            _float_equal(dict(metrics[name]).get("ratio"), 5.0)
            for name in expected_ratios
        ),
        "paired_not_five_times": audit.get(
            "paired_metrics_are_exactly_five_times_iteration_001"
        )
        is False
        and paired.get("baseline_total_after_cost_pnl_both_iterations")
        == -1.09565
        and paired.get("iteration_001_paired_lcb") == -0.1514512500000002
        and paired.get("iteration_003_paired_lcb") == 0.6115974999999999
        and paired.get("paired_lcb_flipped_positive") is True,
        "conclusion": conclusion.get(
            "iteration_003_constitutes_historical_success"
        )
        is False
        and conclusion.get("attempt_002_may_rely_on_iteration_003") is False
        and conclusion.get("pure_sizing_change_can_be_a_future_candidate")
        is False
        and conclusion.get("required_remediation")
        == (
            "evaluate_candidate_and_baseline_at_unit_sizing_for_every_"
            "statistical_gate"
        ),
        "slots": alpha.get("maximum_development_iterations") == 5
        and alpha.get("development_iteration_budget_increased") is False
        and alpha.get("remaining_iteration_slots") == [4, 5]
        and all(alpha.get(f"slot_{slot}_consumed") is True for slot in (1, 2, 3)),
        "safety": governance.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("scale-invariance governance", checks)


def validate_attempt_002_supersession(
    supersession: Mapping[str, Any],
) -> None:
    """Validate the additive attempt-002 supersession before any external use."""

    disposition = dict(supersession.get("disposition") or {})
    lineage = dict(supersession.get("lineage") or {})
    policy = dict(supersession.get("frozen_artifact_policy") or {})
    reason = dict(supersession.get("reason") or {})
    checks = {
        "schema": supersession.get("schema_version")
        == ATTEMPT_002_SUPERSESSION_SCHEMA_VERSION,
        "identity": supersession.get("issue") == 262
        and supersession.get("attempt_id")
        == "v8-1-challenger-future-attempt-002",
        "lineage": lineage.get("attempt_002_preregistration_sha256")
        == "0fa091610966a3a3470872a7e1b5832c8a32985fc312235366ad41aa891f249f"
        and lineage.get("attempt_002_execution_manifest_sha256")
        == "f58e53f317ecdc7de467570b93094034a201a545d50efd48abc54194ea47eabf"
        and lineage.get("scale_invariance_governance_record_sha256")
        == "e8898ef5aa1c4b796109c0d03920d794842472bdfb271f9db7221a100bc8590f",
        "superseded": disposition.get("status")
        == "superseded_before_collection_or_target_access"
        and disposition.get("historical_eligibility_valid_after_governance_review")
        is False
        and disposition.get("promotion_evidence_eligible") is False,
        "nothing_started": all(
            disposition.get(field) is False
            for field in (
                "attempt_consumed_by_target_access",
                "collection_authorized",
                "collection_started",
                "outcomes_resolution_labels_or_pnl_opened",
                "service_root_created",
                "target_access_claim_created",
                "target_access_occurred",
            )
        ),
        "additive": policy.get("additive_supersession_companion") is True
        and policy.get("original_attempt_002_artifacts_rewritten") is False
        and policy.get(
            "supersession_marker_must_be_checked_before_authorization_"
            "collection_target_access_or_promotion"
        )
        is True,
        "reason": reason.get("code")
        == "historical_success_invalidated_by_scale_invariance_gate_defect"
        and reason.get("iteration_003_constitutes_historical_success") is False
        and reason.get(
            "replacement_future_attempt_may_be_preregistered_only_after_"
            "revised_historical_standard_passes"
        )
        is True,
        "safety": supersession.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("attempt-002 supersession", checks)


def validate_historical_development_success_standard(
    standard: Mapping[str, Any],
    *,
    expected_registry_sha256: str,
) -> None:
    """Validate exact preregistered historical success semantics."""

    if standard.get("schema_version") == SUCCESS_STANDARD_V2_SCHEMA_VERSION:
        _validate_historical_development_success_standard_v2(
            standard,
            expected_registry_sha256=expected_registry_sha256,
        )
        return

    corpus = dict(standard.get("development_corpus") or {})
    paired = dict(standard.get("full_window_paired_gate") or {})
    paired_bootstrap = dict(paired.get("bootstrap") or {})
    absolute = dict(standard.get("absolute_candidate_gate") or {})
    absolute_bootstrap = dict(absolute.get("bootstrap") or {})
    robustness = dict(standard.get("robustness_gates") or {})
    split = dict(robustness.get("chronological_split") or {})
    half = dict(robustness.get("half_window_not_significantly_negative") or {})
    half_bootstrap = dict(half.get("bootstrap") or {})
    support = dict(standard.get("support_consistency_gate") or {})
    future = dict(standard.get("future_protocol_alignment") or {})
    concentration = dict(standard.get("concentration_diagnostics") or {})
    discipline = dict(standard.get("iteration_discipline") or {})
    promotion = dict(standard.get("promotion_evidence_policy") or {})
    checks = {
        "schema": standard.get("schema_version") == SUCCESS_STANDARD_SCHEMA_VERSION,
        "issue": standard.get("issue") == 262,
        "frozen": standard.get("frozen") is True
        and standard.get("frozen_before_first_new_development_replay") is True,
        "registry": corpus.get("registry_sha256") == expected_registry_sha256,
        "exact_195": corpus.get("exact_market_count") == 195
        and corpus.get("chronological_order_required") is True
        and corpus.get("market_id_unique_required") is True
        and corpus.get("market_id_sequence_sha256")
        == "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
        and corpus.get("market_id_sequence_sha256_method")
        == "sha256_of_utf8_market_ids_one_per_line_with_final_newline",
        "paired_scope": paired.get("comparison_scope") == "all_195_markets"
        and paired.get("no_trade_after_cost_pnl") == 0.0
        and paired.get("baseline_id") == "matched_frozen_v6_7"
        and paired.get("candidate_id_role")
        == "preregistered_development_candidate",
        "paired_lcb": paired.get(
            "candidate_minus_baseline_after_cost_pnl_bootstrap_lcb_minimum_exclusive"
        )
        == 0.0
        and paired_bootstrap
        == {
            "confidence_level": 0.975,
            "lower_confidence_bound_quantile": 0.025,
            "method": "paired_market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26219501,
            "unit": "market_id",
        },
        "absolute_lcb": absolute.get(
            "candidate_total_after_cost_pnl_bootstrap_lcb_minimum_exclusive"
        )
        == 0.0
        and absolute.get("no_trade_after_cost_pnl") == 0.0
        and absolute_bootstrap
        == {
            "confidence_level": 0.975,
            "lower_confidence_bound_quantile": 0.025,
            "method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26219502,
            "unit": "market_id",
        },
        "largest_winner": robustness.get(
            "candidate_largest_winner_removed_total_after_cost_pnl_minimum_exclusive"
        )
        == 0.0,
        "split": split
        == {
            "first_half_market_count": 97,
            "method": "first_floor_n_over_2_then_remaining",
            "second_half_market_count": 98,
        },
        "half_window_gate": half.get("gate_definition")
        == (
            "one_sided_bootstrap_upper_confidence_bound_"
            "greater_than_or_equal_to_zero"
        )
        and half.get("first_half_seed") == 26219503
        and half.get("second_half_seed") == 26219504
        and half.get("upper_confidence_bound_minimum_inclusive") == 0.0
        and half_bootstrap
        == {
            "confidence_level": 0.975,
            "method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "unit": "market_id",
            "upper_confidence_bound_quantile": 0.975,
        },
        "support_mode": support.get("expected_future_market_count") == 120
        and support.get("future_minimum_accepted_support") is None
        and support.get("future_support_mode")
        == "full_window_paired_no_minimum_accepted_support"
        and future.get("minimum_accepted_support") is None
        and future.get("support_mode")
        == "full_window_paired_no_minimum_accepted_support",
        "support_definition": support.get("gate_definition")
        == (
            "candidate_acceptance_rate_times_future_market_count_must_meet_"
            "future_minimum_support_or_future_protocol_has_no_minimum_support"
        )
        and support.get("report_expected_future_accepted_market_count") is True,
        "future_isomorphic": future.get("full_window_paired_gate_required") is True
        and future.get("absolute_candidate_lcb_gate_required") is True
        and future.get(
            "attempt_002_may_be_preregistered_only_after_all_historical_gates_pass"
        )
        is True
        and future.get(
            "historical_and_future_success_standard_must_be_structurally_isomorphic"
        )
        is True,
        "concentration_diagnostic_only": concentration.get("hard_gate") is False
        and concentration.get("report_selected_side_distribution") is True
        and concentration.get("report_largest_absolute_single_market_pnl_share")
        is True
        and concentration.get("report_selected_action_distribution") is True
        and concentration.get("report_largest_winner_share_of_positive_pnl")
        is True
        and concentration.get("required_side_labels")
        == ["UP", "DOWN", "NONE"],
        "discipline": discipline.get("maximum_development_iterations") == 5
        and discipline.get("candidate_change_preregistration_required_before_evaluation")
        is True
        and discipline.get("alpha_ledger_entry_required_for_every_evaluation")
        is True
        and discipline.get("multiple_candidates_per_iteration_allowed") is False
        and discipline.get("unpreregistered_grid_search_allowed") is False
        and discipline.get("comprehensive_review_required_after_limit") is True
        and discipline.get("candidate_change_preregistration_required_fields")
        == [
            "iteration_number",
            "candidate_id",
            "changed_components",
            "change_description",
            "mechanistic_rationale",
            "expected_mechanism",
            "input_artifact_sha256s",
            "implementation_commit",
        ],
        "historical_never_promotes": promotion.get(
            "historical_development_results_are_promotion_evidence"
        )
        is False
        and promotion.get("historical_pass_can_unlock_attempt_002_preregistration_only")
        is True
        and promotion.get("promotion_evidence_source")
        == "not_yet_collected_future_attempt_002_window_only",
        "safety": standard.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("historical development success standard", checks)


def _validate_historical_development_success_standard_v2(
    standard: Mapping[str, Any],
    *,
    expected_registry_sha256: str,
) -> None:
    """Validate the strictly tighter, scale-invariant iteration-4/5 standard."""

    corpus = dict(standard.get("development_corpus") or {})
    sizing = dict(standard.get("statistical_sizing_policy") or {})
    paired = dict(standard.get("full_window_paired_gate") or {})
    paired_bootstrap = dict(paired.get("bootstrap") or {})
    absolute = dict(standard.get("absolute_candidate_gate") or {})
    absolute_bootstrap = dict(absolute.get("bootstrap") or {})
    robustness = dict(standard.get("robustness_gates") or {})
    split = dict(robustness.get("chronological_split") or {})
    half = dict(robustness.get("half_window_not_significantly_negative") or {})
    half_bootstrap = dict(half.get("bootstrap") or {})
    support = dict(standard.get("support_consistency_gate") or {})
    future = dict(standard.get("future_protocol_alignment") or {})
    concentration = dict(standard.get("concentration_diagnostics") or {})
    discipline = dict(standard.get("iteration_discipline") or {})
    promotion = dict(standard.get("promotion_evidence_policy") or {})
    lineage = dict(standard.get("lineage") or {})
    tightening = dict(standard.get("strict_tightening") or {})
    checks = {
        "schema": standard.get("schema_version")
        == SUCCESS_STANDARD_V2_SCHEMA_VERSION,
        "issue": standard.get("issue") == 262,
        "frozen": standard.get("frozen") is True
        and standard.get("frozen_before_iteration_004_replay") is True,
        "registry": corpus.get("registry_sha256") == expected_registry_sha256,
        "exact_195": corpus.get("exact_market_count") == 195
        and corpus.get("chronological_order_required") is True
        and corpus.get("market_id_unique_required") is True
        and corpus.get("market_id_sequence_sha256")
        == "fef9eda7b8dac138b88c75f96b010bd40953795b2bcf7424debf77a004e06883"
        and corpus.get("market_id_sequence_sha256_method")
        == "sha256_of_utf8_market_ids_one_per_line_with_final_newline",
        "scale_invariant_sizing": sizing
        == {
            "baseline_declared_sizing_is_report_only": True,
            "baseline_unit_pnl_definition": (
                "baseline_after_cost_pnl_divided_by_"
                "baseline_declared_position_size"
            ),
            "candidate_declared_sizing_is_report_only": True,
            "candidate_unit_pnl_definition": (
                "candidate_after_cost_pnl_divided_by_"
                "candidate_declared_position_size"
            ),
            "comparison_rows_must_declare_candidate_and_baseline_position_size": True,
            "declared_size_must_be_finite_and_strictly_positive": True,
            "no_trade_unit_pnl": 0.0,
            "statistical_gate_position_size": 1.0,
            "unit_sizing_applies_to": [
                "full_window_paired_lcb",
                "absolute_candidate_lcb",
                "largest_winner_removed",
                "chronological_half_window_checks",
            ],
        },
        "paired_scope": paired.get("comparison_scope") == "all_195_markets"
        and paired.get("no_trade_after_cost_unit_sizing_pnl") == 0.0
        and paired.get("baseline_id") == "matched_frozen_v6_7"
        and paired.get("candidate_id_role")
        == "preregistered_development_candidate",
        "paired_lcb": paired.get(
            "candidate_minus_baseline_after_cost_unit_sizing_pnl_"
            "bootstrap_lcb_minimum_exclusive"
        )
        == 0.0
        and paired_bootstrap
        == {
            "confidence_level": 0.975,
            "lower_confidence_bound_quantile": 0.025,
            "method": "paired_market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26219501,
            "unit": "market_id",
        },
        "absolute_lcb": absolute.get(
            "candidate_total_after_cost_unit_sizing_pnl_bootstrap_"
            "lcb_minimum_exclusive"
        )
        == 0.0
        and absolute.get("no_trade_after_cost_unit_sizing_pnl") == 0.0
        and absolute_bootstrap
        == {
            "confidence_level": 0.975,
            "lower_confidence_bound_quantile": 0.025,
            "method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "seed": 26219502,
            "unit": "market_id",
        },
        "largest_winner": robustness.get(
            "candidate_largest_winner_removed_total_after_cost_unit_sizing_"
            "pnl_minimum_exclusive"
        )
        == 0.0,
        "split": split
        == {
            "first_half_market_count": 97,
            "method": "first_floor_n_over_2_then_remaining",
            "second_half_market_count": 98,
        },
        "half_window_gate": half.get("gate_definition")
        == (
            "one_sided_bootstrap_upper_confidence_bound_"
            "greater_than_or_equal_to_zero_at_unit_sizing"
        )
        and half.get("first_half_seed") == 26219503
        and half.get("second_half_seed") == 26219504
        and half.get("upper_confidence_bound_minimum_inclusive") == 0.0
        and half_bootstrap
        == {
            "confidence_level": 0.975,
            "method": "market_percentile_bootstrap",
            "resample_count": 10000,
            "unit": "market_id",
            "upper_confidence_bound_quantile": 0.975,
        },
        "support": support
        == {
            "accepted_market_count_minimum_inclusive": 39,
            "accepted_market_rate_minimum_inclusive": 0.2,
            "expected_future_market_count": 120,
            "future_minimum_accepted_support": 24,
            "future_support_mode": (
                "full_window_paired_with_minimum_20_percent_accepted_support"
            ),
            "gate_definition": (
                "accepted_market_count_must_be_at_least_39_of_195"
            ),
            "hard_gate": True,
            "report_expected_future_accepted_market_count": True,
        },
        "future_isomorphic": future.get("full_window_paired_gate_required") is True
        and future.get("absolute_candidate_lcb_gate_required") is True
        and future.get("minimum_accepted_support") == 24
        and future.get("statistical_gate_position_size") == 1.0
        and future.get(
            "replacement_future_attempt_may_be_preregistered_only_after_"
            "all_revised_historical_gates_pass"
        )
        is True
        and future.get(
            "historical_and_future_success_standard_must_be_"
            "structurally_isomorphic"
        )
        is True
        and future.get("superseded_attempt_002_may_be_started") is False,
        "concentration_diagnostic_only": concentration.get("hard_gate") is False
        and concentration.get("report_selected_side_distribution") is True
        and concentration.get(
            "report_largest_absolute_single_market_pnl_share"
        )
        is True
        and concentration.get("report_selected_action_distribution") is True
        and concentration.get("report_largest_winner_share_of_positive_pnl")
        is True
        and concentration.get("required_side_labels")
        == ["UP", "DOWN", "NONE"],
        "discipline": discipline.get("maximum_development_iterations") == 5
        and discipline.get("consumed_iteration_slots") == [1, 2, 3]
        and discipline.get("remaining_iteration_slots") == [4, 5]
        and discipline.get("development_iteration_budget_increased") is False
        and discipline.get("pure_sizing_change_allowed_as_candidate") is False
        and discipline.get(
            "candidate_change_preregistration_required_before_evaluation"
        )
        is True
        and discipline.get("alpha_ledger_entry_required_for_every_evaluation")
        is True
        and discipline.get("multiple_candidates_per_iteration_allowed") is False
        and discipline.get("unpreregistered_grid_search_allowed") is False
        and discipline.get("comprehensive_review_required_after_limit") is True,
        "lineage": lineage
        == {
            "attempt_002_execution_manifest_supersession_sha256": (
                "fdd9b03d3a77343a4310218da2955061d664f0c914c7d979b5ec92a280e32033"
            ),
            "ledger_genesis_frozen_success_standard_sha256": (
                "07609f09692723dd1e650080cfdd29466a7ee8a0f8c30d8378045a8ee3523114"
            ),
            "permitted_previous_iteration_entry_semantic_sha256": (
                "abe1eb3b6e03530c15cb51326801e783454a0be4d5885edd7dcca8dab22779ae"
            ),
            "scale_invariance_governance_record_sha256": (
                "e8898ef5aa1c4b796109c0d03920d794842472bdfb271f9db7221a100bc8590f"
            ),
            "strictly_tightens_success_standard_sha256": (
                "07609f09692723dd1e650080cfdd29466a7ee8a0f8c30d8378045a8ee3523114"
            ),
        },
        "strict_tightening": tightening
        == {
            "prior_frozen_artifacts_rewritten": False,
            "scale_invariance_defect_removed": True,
            "support_gate_added": True,
            "weakened_or_removed_prior_gate": False,
        },
        "historical_never_promotes": promotion.get(
            "historical_development_results_are_promotion_evidence"
        )
        is False
        and promotion.get(
            "historical_pass_can_unlock_replacement_future_attempt_"
            "preregistration_only"
        )
        is True
        and promotion.get("promotion_evidence_source")
        == "not_yet_collected_replacement_future_window_only",
        "safety": standard.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks(
        "historical development success standard v2",
        checks,
    )


def validate_historical_development_ledger_root(
    ledger: Mapping[str, Any],
    *,
    expected_attempt_closure_sha256: str,
    expected_registry_sha256: str,
    expected_success_standard_sha256: str,
) -> None:
    """Validate the immutable genesis for append-only development entries."""

    alpha = dict(ledger.get("alpha_spending_semantics") or {})
    append = dict(ledger.get("append_policy") or {})
    prereg = dict(ledger.get("preregistration_policy") or {})
    prior = {
        str(item.get("evaluation_id")): item
        for item in ledger.get("prior_outcome_aware_evaluations") or []
    }
    checks = {
        "schema": ledger.get("schema_version") == ITERATION_LEDGER_SCHEMA_VERSION,
        "identity": ledger.get("issue") == 262
        and ledger.get("development_family_id")
        == "v8-1-challenge-exact-195-outcome-aware-development"
        and ledger.get("status")
        == "append_only_genesis_ready_for_preregistered_iteration_1"
        and ledger.get("comprehensive_review_required") is False,
        "closure": ledger.get("attempt_001_closure_sha256")
        == expected_attempt_closure_sha256,
        "registry": ledger.get("development_corpus_registry_sha256")
        == expected_registry_sha256,
        "standard": ledger.get("frozen_historical_success_standard_sha256")
        == expected_success_standard_sha256,
        "limit": ledger.get("maximum_development_iterations") == 5
        and ledger.get("next_iteration_number") == 1
        and ledger.get("evaluations_completed_after_standard_freeze") == 0
        and ledger.get("development_iterations") == [],
        "nominal_alpha": alpha.get("nominal_one_sided_alpha_schedule")
        == [0.025] * 5
        and alpha.get("nominal_one_sided_alpha_per_iteration") == 0.025
        and alpha.get("nominal_total_after_five_iterations") == 0.125
        and alpha.get("development_corpus_is_already_outcome_opened") is True
        and alpha.get("each_replay_consumes_one_iteration_slot") is True
        and alpha.get("repeated_development_results_are_promotion_evidence")
        is False
        and alpha.get("confirmatory_type_i_error_claim_allowed") is False
        and alpha.get("fresh_promotion_alpha_consumed") is False,
        "append_only": append.get("ledger_root_rewrite_allowed") is False
        and append.get("entry_files_are_hash_chained") is True
        and append.get("genesis_previous_entry_sha256") == ZERO_SHA256
        and append.get("entry_schema_version") == ITERATION_ENTRY_SCHEMA_VERSION
        and append.get("one_new_entry_file_per_completed_evaluation") is True
        and append.get(
            "preregistration_and_result_are_immutable_entry_descriptors"
        )
        is True,
        "preregistration": prereg.get("candidate_change_must_precede_replay") is True
        and prereg.get("one_candidate_per_iteration") is True
        and prereg.get("preregistration_artifact_sha256_required") is True
        and prereg.get("unpreregistered_grid_search_allowed") is False,
        "prior_evaluations": set(prior)
        == {
            "issue-260-exact-120-historical-replay",
            "issue-252-v6-7-exact-195-replay",
            "issue-260-v8-1-exact-195-diagnostic",
        }
        and all(
            item.get("performed_before_success_standard_freeze") is True
            and item.get("promotion_evidence_eligible") is False
            for item in prior.values()
        )
        and prior["issue-260-exact-120-historical-replay"].get("report_sha256")
        == "558a1802513e70426d9bd4c589b2e7736ea0ed11a704f51fdaa613e18ef45988"
        and prior["issue-252-v6-7-exact-195-replay"].get("report_sha256")
        == "500b51146f7d9fc0c7284702ba954fd4aac2d36644e8b18f1f61121deb1b0ff4"
        and prior["issue-260-v8-1-exact-195-diagnostic"].get("report_sha256")
        == "8d5458ee09a3a4fa28a94042db947caaf8dbe3c0874cf96e0435d34bf20d179a",
        "safety": ledger.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("historical development ledger root", checks)


def validate_iteration_preregistration(
    preregistration: Mapping[str, Any],
    *,
    expected_iteration_number: int,
    expected_candidate_id: str,
    expected_registry_sha256: str,
    expected_success_standard_sha256: str,
    expected_ledger_root_sha256: str,
    expected_previous_entry_sha256: str,
    expected_implementation_base_commit: str,
    success_standard: Mapping[str, Any] | None = None,
) -> None:
    """Require a concrete, one-candidate rationale before outcome-aware replay."""

    changed = list(preregistration.get("changed_components") or [])
    inputs = dict(preregistration.get("input_artifact_sha256s") or {})
    scale_invariant_standard = (
        success_standard is not None
        and success_standard.get("schema_version")
        == SUCCESS_STANDARD_V2_SCHEMA_VERSION
    )
    checks = {
        "schema": preregistration.get("schema_version")
        == ITERATION_PREREGISTRATION_SCHEMA_VERSION,
        "iteration": preregistration.get("iteration_number")
        == expected_iteration_number
        and 1 <= expected_iteration_number <= 5,
        "candidate": preregistration.get("candidate_id") == expected_candidate_id,
        "registry": preregistration.get("development_corpus_registry_sha256")
        == expected_registry_sha256,
        "standard": preregistration.get("success_standard_sha256")
        == expected_success_standard_sha256,
        "ledger": preregistration.get("ledger_root_sha256")
        == expected_ledger_root_sha256
        and preregistration.get("previous_iteration_entry_sha256")
        == expected_previous_entry_sha256,
        "changed_components": bool(changed)
        and len(changed) == len(set(changed))
        and set(changed) <= {"controller", "threshold", "feature", "sizing"},
        "not_pure_sizing": not scale_invariant_standard or changed != ["sizing"],
        "rationale": all(
            isinstance(preregistration.get(field), str)
            and bool(str(preregistration.get(field)).strip())
            for field in (
                "change_description",
                "mechanistic_rationale",
                "expected_mechanism",
                "implementation_commit",
            )
        )
        and preregistration.get("implementation_commit")
        == expected_implementation_base_commit
        and preregistration.get("implementation_commit_role")
        == "prechange_base_commit",
        "inputs": bool(inputs)
        and all(_is_sha256(value) for value in inputs.values()),
        "one_candidate": preregistration.get("candidate_count") == 1,
        "no_grid": preregistration.get("grid_search") is False
        and preregistration.get("result_selected_parameter_search") is False,
        "before_replay": preregistration.get("evaluation_started") is False
        and preregistration.get("outcome_aware_replay_started") is False,
        "historical_only": preregistration.get("promotion_evidence_eligible") is False,
        "safety": preregistration.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("historical development preregistration", checks)


def evaluate_historical_development_candidate(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    success_standard: Mapping[str, Any],
    candidate_id: str,
    iteration_number: int,
) -> dict[str, Any]:
    """Evaluate exactly one candidate on all 195 registered development markets."""

    if not candidate_id or not 1 <= iteration_number <= 5:
        raise ChallengeHistoricalDevelopmentError(
            "candidate_id and iteration_number 1-5 are required"
        )
    registry_sha = str(
        (success_standard.get("development_corpus") or {}).get("registry_sha256")
        or ""
    )
    validate_historical_development_success_standard(
        success_standard,
        expected_registry_sha256=registry_sha,
    )
    scale_invariant_standard = (
        success_standard.get("schema_version")
        == SUCCESS_STANDARD_V2_SCHEMA_VERSION
    )
    normalized = [
        _normalize_comparison_row(
            row,
            index,
            require_position_sizes=scale_invariant_standard,
        )
        for index, row in enumerate(comparison_rows)
    ]
    expected_count = int(success_standard["development_corpus"]["exact_market_count"])
    if len(normalized) != expected_count:
        raise ChallengeHistoricalDevelopmentError(
            f"comparison row count must be exactly {expected_count}"
        )
    market_ids = [row["market_id"] for row in normalized]
    if len(set(market_ids)) != len(market_ids):
        raise ChallengeHistoricalDevelopmentError("comparison market_id values are not unique")
    expected_sequence_sha256 = str(
        success_standard["development_corpus"]["market_id_sequence_sha256"]
    )
    actual_sequence_sha256 = _market_id_sequence_sha256(market_ids)
    if actual_sequence_sha256 != expected_sequence_sha256:
        raise ChallengeHistoricalDevelopmentError(
            "comparison market_id sequence does not match the frozen exact-195 corpus"
        )

    candidate_declared_pnl = [
        row["candidate_after_cost_pnl"] for row in normalized
    ]
    baseline_declared_pnl = [
        row["baseline_after_cost_pnl"] for row in normalized
    ]
    candidate_pnl = [
        (
            row["candidate_unit_after_cost_pnl"]
            if scale_invariant_standard
            else row["candidate_after_cost_pnl"]
        )
        for row in normalized
    ]
    baseline_pnl = [
        (
            row["baseline_unit_after_cost_pnl"]
            if scale_invariant_standard
            else row["baseline_after_cost_pnl"]
        )
        for row in normalized
    ]
    paired_delta = [
        candidate - baseline
        for candidate, baseline in zip(candidate_pnl, baseline_pnl, strict=True)
    ]
    paired_spec = success_standard["full_window_paired_gate"]["bootstrap"]
    absolute_spec = success_standard["absolute_candidate_gate"]["bootstrap"]
    paired_distribution = _bootstrap_sums(
        paired_delta,
        resample_count=int(paired_spec["resample_count"]),
        seed=int(paired_spec["seed"]),
    )
    absolute_distribution = _bootstrap_sums(
        candidate_pnl,
        resample_count=int(absolute_spec["resample_count"]),
        seed=int(absolute_spec["seed"]),
    )
    paired_lcb = _quantile(
        paired_distribution,
        float(paired_spec["lower_confidence_bound_quantile"]),
    )
    absolute_lcb = _quantile(
        absolute_distribution,
        float(absolute_spec["lower_confidence_bound_quantile"]),
    )

    split_spec = success_standard["robustness_gates"]["chronological_split"]
    first_count = int(split_spec["first_half_market_count"])
    first_pnl = candidate_pnl[:first_count]
    second_pnl = candidate_pnl[first_count:]
    half_spec = success_standard["robustness_gates"][
        "half_window_not_significantly_negative"
    ]
    half_bootstrap = half_spec["bootstrap"]
    first_distribution = _bootstrap_sums(
        first_pnl,
        resample_count=int(half_bootstrap["resample_count"]),
        seed=int(half_spec["first_half_seed"]),
    )
    second_distribution = _bootstrap_sums(
        second_pnl,
        resample_count=int(half_bootstrap["resample_count"]),
        seed=int(half_spec["second_half_seed"]),
    )
    half_quantile = float(half_bootstrap["upper_confidence_bound_quantile"])
    first_ucb = _quantile(first_distribution, half_quantile)
    second_ucb = _quantile(second_distribution, half_quantile)

    largest_winner = max(candidate_pnl)
    largest_winner_removed = sum(candidate_pnl) - largest_winner
    accepted = [row for row in normalized if row["candidate_action"] != "NO_TRADE"]
    acceptance_rate = len(accepted) / len(normalized)
    support_spec = success_standard["support_consistency_gate"]
    future_count = int(support_spec["expected_future_market_count"])
    expected_future_support = acceptance_rate * future_count
    future_minimum = support_spec["future_minimum_accepted_support"]
    if scale_invariant_standard:
        minimum_accepted = int(
            support_spec["accepted_market_count_minimum_inclusive"]
        )
        minimum_rate = float(
            support_spec["accepted_market_rate_minimum_inclusive"]
        )
        support_passed = (
            len(accepted) >= minimum_accepted
            and acceptance_rate >= minimum_rate
            and expected_future_support >= float(future_minimum)
        )
    else:
        minimum_accepted = None
        support_passed = (
            future_minimum is None
            or expected_future_support >= float(future_minimum)
        )
    side_distribution = Counter(row["candidate_side"] for row in normalized)
    action_distribution = Counter(row["candidate_action"] for row in normalized)
    absolute_pnl_sum = sum(abs(value) for value in candidate_pnl)
    positive_pnl_sum = sum(max(value, 0.0) for value in candidate_pnl)
    largest_absolute_share = (
        max(abs(value) for value in candidate_pnl) / absolute_pnl_sum
        if absolute_pnl_sum
        else 0.0
    )
    largest_winner_share = largest_winner / positive_pnl_sum if positive_pnl_sum else 0.0

    checks = {
        "exact_195_unique_frozen_chronological_market_rows": (
            len(normalized) == expected_count
            and actual_sequence_sha256 == expected_sequence_sha256
        ),
        "full_window_paired_bootstrap_97_5_lcb_positive": paired_lcb > 0.0,
        "candidate_absolute_bootstrap_97_5_lcb_positive": absolute_lcb > 0.0,
        "candidate_largest_winner_removed_pnl_positive": largest_winner_removed > 0.0,
        "first_half_not_significantly_negative": first_ucb >= 0.0,
        "second_half_not_significantly_negative": second_ucb >= 0.0,
        "support_consistent_with_future_protocol": support_passed,
        "accepted_market_count_at_least_39": (
            len(accepted) >= minimum_accepted
            if minimum_accepted is not None
            else True
        ),
        "statistical_gates_use_unit_sizing": scale_invariant_standard
        or success_standard.get("schema_version")
        == SUCCESS_STANDARD_SCHEMA_VERSION,
        "historical_results_not_promotion_evidence": True,
        "all_safety_unlocks_remain_false": True,
    }
    all_passed = all(checks.values())
    return {
        "schema_version": ITERATION_RESULT_SCHEMA_VERSION,
        "iteration_number": iteration_number,
        "candidate_id": candidate_id,
        "development_corpus_id": success_standard["development_corpus"][
            "registered_corpus_id"
        ],
        "market_count": len(normalized),
        "metrics": {
            "accepted_market_count": len(accepted),
            "acceptance_rate": acceptance_rate,
            "expected_accepted_markets_in_120": expected_future_support,
            "statistical_gate_position_size": (
                1.0 if scale_invariant_standard else None
            ),
            "candidate_total_after_cost_pnl": sum(candidate_declared_pnl),
            "baseline_total_after_cost_pnl": sum(baseline_declared_pnl),
            "candidate_minus_baseline_total_after_cost_pnl": sum(
                candidate - baseline
                for candidate, baseline in zip(
                    candidate_declared_pnl,
                    baseline_declared_pnl,
                    strict=True,
                )
            ),
            "candidate_unit_sizing_total_after_cost_pnl": sum(candidate_pnl),
            "baseline_unit_sizing_total_after_cost_pnl": sum(baseline_pnl),
            "candidate_minus_baseline_unit_sizing_total_after_cost_pnl": sum(
                paired_delta
            ),
            "candidate_largest_winner_after_cost_pnl": largest_winner,
            "candidate_largest_winner_removed_after_cost_pnl": largest_winner_removed,
            "first_half_candidate_total_after_cost_pnl": sum(first_pnl),
            "second_half_candidate_total_after_cost_pnl": sum(second_pnl),
        },
        "bootstrap": {
            "paired_delta_97_5_lcb": paired_lcb,
            "candidate_absolute_97_5_lcb": absolute_lcb,
            "first_half_candidate_97_5_ucb": first_ucb,
            "second_half_candidate_97_5_ucb": second_ucb,
        },
        "concentration_diagnostics": {
            "hard_gate": False,
            "selected_side_distribution": dict(sorted(side_distribution.items())),
            "selected_action_distribution": dict(sorted(action_distribution.items())),
            "largest_absolute_single_market_pnl_share": largest_absolute_share,
            "largest_winner_share_of_positive_pnl": largest_winner_share,
        },
        "checks": checks,
        "all_historical_success_criteria_passed": all_passed,
        "attempt_002_preregistration_allowed": (
            all_passed if not scale_invariant_standard else False
        ),
        "replacement_future_attempt_preregistration_allowed": (
            all_passed if scale_invariant_standard else False
        ),
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "safety": SAFE_FALSES,
    }


def run_historical_development_evaluation(
    config: HistoricalDevelopmentEvaluationConfig,
) -> dict[str, Any]:
    """Run one preregistered evaluation and emit immutable result descriptors."""

    closure = _load_pinned_json(
        config.attempt_closure_path,
        config.expected_attempt_closure_sha256,
        label="attempt closure",
    )
    registry = _load_pinned_json(
        config.registry_path,
        config.expected_registry_sha256,
        label="development registry",
    )
    standard = _load_pinned_json(
        config.success_standard_path,
        config.expected_success_standard_sha256,
        label="success standard",
    )
    ledger = _load_pinned_json(
        config.ledger_root_path,
        config.expected_ledger_root_sha256,
        label="ledger root",
    )
    preregistration = _load_pinned_json(
        config.preregistration_path,
        config.expected_preregistration_sha256,
        label="iteration preregistration",
    )
    validate_attempt_001_closure(
        closure,
        expected_collection_plan_sha256=FROZEN_COLLECTION_PLAN_SHA256,
        expected_supersession_governance_sha256=(
            FROZEN_SUPERSESSION_GOVERNANCE_SHA256
        ),
    )
    validate_historical_development_registry(
        registry,
        expected_attempt_closure_sha256=config.expected_attempt_closure_sha256,
        expected_evidence_ledger_sha256=(
            "98f43a1a9526d9b21342c6047aaf6ca34f78e0149a0c31adbd58fd7e98a11bf3"
        ),
    )
    validate_historical_development_success_standard(
        standard,
        expected_registry_sha256=config.expected_registry_sha256,
    )
    validate_historical_development_ledger_root(
        ledger,
        expected_attempt_closure_sha256=config.expected_attempt_closure_sha256,
        expected_registry_sha256=config.expected_registry_sha256,
        expected_success_standard_sha256=(
            str(
                (standard.get("lineage") or {}).get(
                    "ledger_genesis_frozen_success_standard_sha256"
                )
            )
            if standard.get("schema_version")
            == SUCCESS_STANDARD_V2_SCHEMA_VERSION
            else config.expected_success_standard_sha256
        ),
    )
    validate_iteration_preregistration(
        preregistration,
        expected_iteration_number=config.iteration_number,
        expected_candidate_id=config.candidate_id,
        expected_registry_sha256=config.expected_registry_sha256,
        expected_success_standard_sha256=config.expected_success_standard_sha256,
        expected_ledger_root_sha256=config.expected_ledger_root_sha256,
        expected_previous_entry_sha256=config.previous_iteration_entry_sha256,
        expected_implementation_base_commit=config.implementation_base_commit,
        success_standard=standard,
    )
    previous_entry = _validate_previous_iteration_entry(
        config,
        success_standard=standard,
    )
    _verify_sha256(
        config.comparison_rows_path,
        config.expected_comparison_rows_sha256,
        label="comparison rows",
    )
    run_dir = config.output_dir.resolve() / config.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(config.comparison_rows_path)
    report = evaluate_historical_development_candidate(
        rows,
        success_standard=standard,
        candidate_id=config.candidate_id,
        iteration_number=config.iteration_number,
    )
    report.update(
        {
            "run_id": config.run_id,
            "implementation_base_commit": config.implementation_base_commit,
            "preregistration_commit": config.preregistration_commit,
            "implementation_commit": config.implementation_commit,
            "evaluated_at": config.evaluated_at,
            "input_descriptors": {
                "attempt_closure": _descriptor(config.attempt_closure_path),
                "development_registry": _descriptor(config.registry_path),
                "success_standard": _descriptor(config.success_standard_path),
                "ledger_root": _descriptor(config.ledger_root_path),
                "preregistration": _descriptor(config.preregistration_path),
                "comparison_rows": _descriptor(config.comparison_rows_path),
            },
        }
    )
    if config.previous_iteration_entry_path is not None:
        report["input_descriptors"]["previous_iteration_entry"] = _descriptor(
            config.previous_iteration_entry_path
        )
    report_path = run_dir / "challenge_historical_development_result.json"
    _write_json(report_path, report)
    report_sha256 = _sha256_file(report_path)
    entry = {
        "schema_version": ITERATION_ENTRY_SCHEMA_VERSION,
        "sequence": config.iteration_number,
        "entry_id": f"challenge-historical-development-entry-{config.iteration_number:03d}",
        "previous_entry_sha256": config.previous_iteration_entry_sha256,
        "implementation_base_commit": config.implementation_base_commit,
        "preregistration_commit": config.preregistration_commit,
        "implementation_commit": config.implementation_commit,
        "ledger_root_sha256": config.expected_ledger_root_sha256,
        "development_registry_sha256": config.expected_registry_sha256,
        "success_standard_sha256": config.expected_success_standard_sha256,
        "candidate_id": config.candidate_id,
        "preregistration": _descriptor(config.preregistration_path),
        "result": {"path": str(report_path), "sha256": report_sha256},
        "nominal_one_sided_alpha": 0.025,
        "consumes_development_iteration_slot": True,
        "consumes_fresh_promotion_alpha": False,
        "historical_result_is_promotion_evidence": False,
        "all_historical_success_criteria_passed": report[
            "all_historical_success_criteria_passed"
        ],
        "safety": SAFE_FALSES,
    }
    entry["entry_sha256"] = _semantic_sha256(entry)
    if previous_entry is not None and previous_entry["sequence"] != config.iteration_number - 1:
        raise ChallengeHistoricalDevelopmentError(
            "previous iteration entry sequence is not contiguous"
        )
    entry_path = run_dir / "challenge_historical_development_iteration_entry.json"
    _write_json(entry_path, entry)
    manifest = {
        "schema_version": (
            "bigan-v8-challenge-historical-development-evaluation-manifest-v1"
        ),
        "run_id": config.run_id,
        "iteration_number": config.iteration_number,
        "candidate_id": config.candidate_id,
        "report": {"path": str(report_path), "sha256": report_sha256},
        "iteration_entry": _descriptor(entry_path),
        "historical_development_only": True,
        "promotion_evidence_eligible": False,
        "attempt_002_preregistration_allowed": report[
            "attempt_002_preregistration_allowed"
        ],
        "replacement_future_attempt_preregistration_allowed": report[
            "replacement_future_attempt_preregistration_allowed"
        ],
        "safety": SAFE_FALSES,
    }
    manifest_path = run_dir / "challenge_historical_development_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": report_sha256,
        "iteration_entry": entry,
        "iteration_entry_path": entry_path,
        "iteration_entry_sha256": entry["entry_sha256"],
        "iteration_entry_file_sha256": _sha256_file(entry_path),
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _normalize_comparison_row(
    row: Mapping[str, Any],
    chronological_index: int,
    *,
    require_position_sizes: bool = False,
) -> dict[str, Any]:
    market_id = str(row.get("market_id") or "")
    if not market_id:
        raise ChallengeHistoricalDevelopmentError("comparison market_id is missing")
    candidate = _finite_float(
        row.get(
            "candidate_after_cost_pnl",
            row.get("challenge_after_cost_pnl"),
        ),
        field="candidate_after_cost_pnl",
    )
    baseline = _finite_float(
        row.get(
            "baseline_after_cost_pnl",
            row.get("v6_7_after_cost_pnl"),
        ),
        field="baseline_after_cost_pnl",
    )
    action = str(row.get("candidate_action", row.get("challenge_action")) or "")
    side = str(row.get("candidate_side", row.get("challenge_side")) or "")
    baseline_action = str(
        row.get("baseline_action", row.get("v6_7_action")) or ""
    )
    baseline_side = str(
        row.get("baseline_side", row.get("v6_7_side")) or _side_for_action(baseline_action)
    )
    if side != _side_for_action(action):
        raise ChallengeHistoricalDevelopmentError(
            "comparison candidate action or side is invalid"
        )
    if baseline_side != _side_for_action(baseline_action):
        raise ChallengeHistoricalDevelopmentError(
            "comparison baseline action or side is invalid"
        )
    if action == "NO_TRADE" and not _float_equal(candidate, 0.0):
        raise ChallengeHistoricalDevelopmentError("NO_TRADE must have NONE side and zero PnL")
    if baseline_action == "NO_TRADE" and not _float_equal(baseline, 0.0):
        raise ChallengeHistoricalDevelopmentError(
            "baseline NO_TRADE must have NONE side and zero PnL"
        )
    supplied_delta = row.get(
        "candidate_minus_baseline_pnl",
        row.get("challenge_minus_v6_7_pnl"),
    )
    if supplied_delta is not None and not _float_equal(
        supplied_delta,
        candidate - baseline,
    ):
        raise ChallengeHistoricalDevelopmentError("paired market PnL delta does not reconcile")
    candidate_size_value = row.get("candidate_declared_position_size")
    baseline_size_value = row.get("baseline_declared_position_size")
    if require_position_sizes and (
        candidate_size_value is None or baseline_size_value is None
    ):
        raise ChallengeHistoricalDevelopmentError(
            "scale-invariant comparison rows must declare candidate and "
            "baseline position size"
        )
    candidate_size = (
        _positive_finite_float(
            candidate_size_value,
            field="candidate_declared_position_size",
        )
        if candidate_size_value is not None
        else 1.0
    )
    baseline_size = (
        _positive_finite_float(
            baseline_size_value,
            field="baseline_declared_position_size",
        )
        if baseline_size_value is not None
        else 1.0
    )
    candidate_unit_pnl = candidate / candidate_size
    baseline_unit_pnl = baseline / baseline_size
    supplied_candidate_unit = row.get("candidate_unit_after_cost_pnl")
    supplied_baseline_unit = row.get("baseline_unit_after_cost_pnl")
    if supplied_candidate_unit is not None and not _float_equal(
        supplied_candidate_unit,
        candidate_unit_pnl,
    ):
        raise ChallengeHistoricalDevelopmentError(
            "candidate unit-sizing PnL does not reconcile"
        )
    if supplied_baseline_unit is not None and not _float_equal(
        supplied_baseline_unit,
        baseline_unit_pnl,
    ):
        raise ChallengeHistoricalDevelopmentError(
            "baseline unit-sizing PnL does not reconcile"
        )
    return {
        "chronological_index": chronological_index,
        "market_id": market_id,
        "candidate_action": action,
        "candidate_side": side,
        "candidate_after_cost_pnl": candidate,
        "baseline_after_cost_pnl": baseline,
        "baseline_action": baseline_action,
        "baseline_side": baseline_side,
        "candidate_declared_position_size": candidate_size,
        "baseline_declared_position_size": baseline_size,
        "candidate_unit_after_cost_pnl": candidate_unit_pnl,
        "baseline_unit_after_cost_pnl": baseline_unit_pnl,
    }


def _side_for_action(action: str) -> str:
    if action == "NO_TRADE":
        return "NONE"
    if action.startswith("BUY_UP_"):
        return "UP"
    if action.startswith("BUY_DOWN_"):
        return "DOWN"
    return ""


def _validate_previous_iteration_entry(
    config: HistoricalDevelopmentEvaluationConfig,
    *,
    success_standard: Mapping[str, Any],
) -> dict[str, Any] | None:
    if config.iteration_number == 1:
        return None
    path = config.previous_iteration_entry_path
    if path is None:
        raise ChallengeHistoricalDevelopmentError(
            "previous iteration entry path is required"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeHistoricalDevelopmentError(
            "previous iteration entry must be a JSON object"
        )
    expected = config.previous_iteration_entry_sha256
    same_standard = (
        payload.get("success_standard_sha256")
        == config.expected_success_standard_sha256
    )
    v2_bridge = (
        config.iteration_number == 4
        and success_standard.get("schema_version")
        == SUCCESS_STANDARD_V2_SCHEMA_VERSION
        and payload.get("entry_sha256")
        == (success_standard.get("lineage") or {}).get(
            "permitted_previous_iteration_entry_semantic_sha256"
        )
        and payload.get("success_standard_sha256")
        == (success_standard.get("lineage") or {}).get(
            "strictly_tightens_success_standard_sha256"
        )
    )
    checks = {
        "schema": payload.get("schema_version") == ITERATION_ENTRY_SCHEMA_VERSION,
        "semantic_hash": payload.get("entry_sha256") == expected
        and _semantic_sha256(payload) == expected,
        "sequence": payload.get("sequence") == config.iteration_number - 1,
        "ledger": payload.get("ledger_root_sha256")
        == config.expected_ledger_root_sha256,
        "registry": payload.get("development_registry_sha256")
        == config.expected_registry_sha256,
        "standard": same_standard or v2_bridge,
        "slot_consumed": payload.get("consumes_development_iteration_slot") is True,
        "historical_only": payload.get("historical_result_is_promotion_evidence")
        is False
        and payload.get("consumes_fresh_promotion_alpha") is False,
        "safety": payload.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("previous historical development entry", checks)
    return payload


def _market_id_sequence_sha256(market_ids: Sequence[str]) -> str:
    payload = "".join(f"{market_id}\n" for market_id in market_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bootstrap_sums(
    values: Sequence[float],
    *,
    resample_count: int,
    seed: int,
) -> list[float]:
    if not values or resample_count <= 0:
        raise ChallengeHistoricalDevelopmentError("bootstrap inputs are invalid")
    count = len(values)
    if all(_float_equal(value, values[0]) for value in values[1:]):
        return [values[0] * count] * resample_count
    rng = random.Random(seed)
    return [
        sum(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resample_count)
    ]


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ChallengeHistoricalDevelopmentError("quantile inputs are invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_pinned_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    _verify_sha256(path, expected_sha256, label=label)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ChallengeHistoricalDevelopmentError(f"{label} must be a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ChallengeHistoricalDevelopmentError(
                f"comparison row {line_number} must be a JSON object"
            )
        rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_sha256(path: Path, expected_sha256: str, *, label: str) -> None:
    _require_sha256(expected_sha256, field=f"{label} SHA-256")
    actual = _sha256_file(path)
    if actual != expected_sha256.lower():
        raise ChallengeHistoricalDevelopmentError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_sha256(payload: Mapping[str, Any]) -> str:
    content = {
        key: value
        for key, value in payload.items()
        if key not in {"entry_sha256"}
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _raise_failed_checks(label: str, checks: Mapping[str, bool]) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeHistoricalDevelopmentError(
            f"{label} invalid: {','.join(blockers)}"
        )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ChallengeHistoricalDevelopmentError(f"{field} is not numeric") from error
    if not math.isfinite(number):
        raise ChallengeHistoricalDevelopmentError(f"{field} must be finite")
    return number


def _positive_finite_float(value: Any, *, field: str) -> float:
    number = _finite_float(value, field=field)
    if number <= 0.0:
        raise ChallengeHistoricalDevelopmentError(
            f"{field} must be strictly positive"
        )
    return number


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_sha256(value: Any, *, field: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a SHA-256 digest")


__all__ = [
    "ATTEMPT_002_SUPERSESSION_SCHEMA_VERSION",
    "ATTEMPT_CLOSURE_SCHEMA_VERSION",
    "ChallengeHistoricalDevelopmentError",
    "DEVELOPMENT_REGISTRY_SCHEMA_VERSION",
    "FROZEN_COLLECTION_PLAN_SHA256",
    "FROZEN_SUPERSESSION_GOVERNANCE_SHA256",
    "HistoricalDevelopmentEvaluationConfig",
    "ITERATION_ENTRY_SCHEMA_VERSION",
    "ITERATION_LEDGER_SCHEMA_VERSION",
    "ITERATION_PREREGISTRATION_SCHEMA_VERSION",
    "ITERATION_RESULT_SCHEMA_VERSION",
    "SUCCESS_STANDARD_SCHEMA_VERSION",
    "SUCCESS_STANDARD_V2_SCHEMA_VERSION",
    "SCALE_INVARIANCE_GOVERNANCE_SCHEMA_VERSION",
    "ZERO_SHA256",
    "evaluate_historical_development_candidate",
    "run_historical_development_evaluation",
    "validate_attempt_001_closure",
    "validate_attempt_002_supersession",
    "validate_historical_development_ledger_root",
    "validate_historical_development_registry",
    "validate_historical_development_success_standard",
    "validate_iteration_preregistration",
    "validate_scale_invariance_governance",
]
