"""Single-use exact-120 future PnL gate for the retained v6.7 policy (#238)."""

from __future__ import annotations

import math
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_direct_advantage_estimand_audit import (
    _market_bootstrap_interval,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    SBC_ACTIONS,
    SIDES,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _find_nonempty_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_post_freeze import (
    _write_single_use_claim,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
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
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)

SCHEMA_PREFIX = "bigan-v8-retained-v6-7-future-noninferiority-v7-5"
PROFILE_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-profile-v1"
TARGET_FREE_FREEZE_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1"
SETTLED_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-settled-corpus-index-v1"
CANDIDATE_NAME = "v7_5_retained_v6_7_policy"
BASELINE_NAME = "v6_7_frozen_policy"
WINDOW_MARKET_COUNT = 120
FIVE_ACTIONS = frozenset(
    {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    }
)
STAGES = {"freeze_target_free", "settle", "evaluate_future_pnl"}
FROZEN_PROFILE_SHA256 = "9f4fc37f613d425e61a03ddb1fe87851477759ada900d9ef5b62a96727c3cf79"
FROZEN_RUNTIME_POLICY_PROFILE_SHA256 = (
    "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
)
FORBIDDEN_TARGET_FIELDS = frozenset(
    {
        "outcome",
        "resolved_outcome",
        "resolution",
        "winner",
        "settlement_pnl",
        "settlement_price",
        "runtime_policy_after_cost_net_pnl_per_contract",
        "runtime_policy_after_cost_net_pnl_at_frozen_size",
        "realized_pnl",
        "realized_return",
        "future_return",
        "label",
        "oracle_action",
    }
)


def _safety_fields() -> dict[str, Any]:
    return {**_blocked_safety_fields(), "paper_candidate_allowed": False}


@dataclass(frozen=True, slots=True)
class RetainedV67FutureNoninferiorityConfig:
    """Pinned inputs for one #238 target-free, settlement, or evaluation stage."""

    stage: Literal["freeze_target_free", "settle", "evaluate_future_pnl"]
    run_id: str
    output_dir: Path | str
    profile_path: Path | str
    expected_profile_sha256: str
    source_freeze_manifest_path: Path | str
    expected_source_freeze_manifest_sha256: str
    implementation_commit: str
    stage_started_ts: int
    target_free_freeze_manifest_path: Path | str | None = None
    expected_target_free_freeze_manifest_sha256: str | None = None
    runtime_policy_profile_path: Path | str | None = None
    expected_runtime_policy_profile_sha256: str | None = None
    settled_corpus_index_path: Path | str | None = None
    expected_settled_corpus_index_sha256: str | None = None
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if self.stage not in STAGES or not self.run_id.strip() or self.stage_started_ts <= 0:
            raise ValueError("#238 stage, run_id, and stage timestamp are required")
        _require_git_sha(self.implementation_commit)
        _require_sha256(self.expected_profile_sha256, name="expected_profile_sha256")
        _require_sha256(
            self.expected_source_freeze_manifest_sha256,
            name="expected_source_freeze_manifest_sha256",
        )
        if self.stage in {"settle", "evaluate_future_pnl"}:
            for name in (
                "target_free_freeze_manifest_path",
                "expected_target_free_freeze_manifest_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#238 target-free input missing: {name}")
            _require_sha256(
                str(self.expected_target_free_freeze_manifest_sha256),
                name="expected_target_free_freeze_manifest_sha256",
            )
        if self.stage == "evaluate_future_pnl":
            for name in (
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#238 evaluation input missing: {name}")
            _require_sha256(
                str(self.expected_runtime_policy_profile_sha256),
                name="expected_runtime_policy_profile_sha256",
            )
            _require_sha256(
                str(self.expected_settled_corpus_index_sha256),
                name="expected_settled_corpus_index_sha256",
            )
        for name in (
            "output_dir",
            "profile_path",
            "source_freeze_manifest_path",
            "target_free_freeze_manifest_path",
            "runtime_policy_profile_path",
            "settled_corpus_index_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


def validate_retained_v6_7_future_profile(profile: dict[str, Any]) -> None:
    """Reject comparison, lineage, target-access, or safety drift."""

    retained = dict(profile.get("retained_policy_contract") or {})
    window = dict(profile.get("future_window") or {})
    gate = dict(profile.get("future_pnl_gate") or {})
    access = dict(profile.get("target_access_contract") or {})
    downstream = dict(profile.get("downstream_contract") or {})
    checks = {
        "schema": profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "issue": profile.get("issue_number") == 238,
        "candidate": profile.get("candidate_name") == CANDIDATE_NAME,
        "frozen": profile.get("frozen") is True,
        "retained_policy": retained
        == {
            "candidate_policy": CANDIDATE_NAME,
            "baseline_policy": BASELINE_NAME,
            "candidate_decisions_are_byte_identical_to_baseline": True,
            "historical_noninferiority_gate_passed": True,
            "model_improvement_demonstrated": False,
            "score_mutation_allowed": False,
            "action_mutation_allowed": False,
            "side_mutation_allowed": False,
            "decision_timestamp_mutation_allowed": False,
            "cost_sizing_or_guard_mutation_allowed": False,
        },
        "window": window
        == {
            "role": "future_confirmatory",
            "exact_market_count": WINDOW_MARKET_COUNT,
            "one_frozen_decision_per_market": True,
            "all_markets_closed_before_target_access": True,
            "complete_five_action_grid_required": True,
            "feature_causality_required": True,
            "strictly_later_and_disjoint_already_frozen": True,
            "side_composition_is_regime_emergent": True,
            "side_quota_enabled": False,
            "side_pnl_hard_gate_enabled": False,
        },
        "inclusive_noninferiority": (
            gate.get("candidate_minus_v6_7_after_cost_pnl_minimum_inclusive") == 0.0
            and gate.get("candidate_minus_v6_7_largest_winner_removed_minimum_inclusive")
            == 0.0
            and gate.get("candidate_minus_v6_7_bootstrap_lcb_minimum_inclusive") == 0.0
            and gate.get("comparison_operator") == "greater_than_or_equal"
            and gate.get("equality_passes_noninferiority") is True
            and gate.get("model_improvement_reported_separately") is True
        ),
        "absolute_pnl_gates": (
            gate.get("minimum_guard_accepted_unique_market_count") == 40
            and gate.get("accepted_total_after_cost_pnl_minimum_exclusive") == 0.0
            and gate.get("largest_winner_removed_after_cost_pnl_minimum_exclusive") == 0.0
            and gate.get("bootstrap_unit") == "market_id"
            and gate.get("bootstrap_confidence_level") == 0.95
            and gate.get("bootstrap_resample_count") == 5000
            and gate.get("bootstrap_seed") == 2382026
            and gate.get("side_action_and_family_metrics_diagnostic_only") is True
            and gate.get("single_use_future_gate") is True
            and gate.get("result_selected_rerun_allowed") is False
            and gate.get("result_selected_extension_allowed") is False
        ),
        "target_access": access
        == {
            "target_free_freeze_before_settlement": True,
            "official_read_only_settlement_on_quarantine_copies": True,
            "source_outcome_blind_artifacts_mutated": False,
            "complete_settlement_required": True,
            "outcomes_used_for_selection_or_tuning": False,
            "target_used_as_decision_time_input": False,
        },
        "downstream": downstream
        == {
            "bounded_paper_candidate_review_requires_future_pnl_gate": True,
            "paper_candidate_auto_unlock_allowed": False,
            "future_pnl_pass_does_not_enable_live_or_handoff": True,
        },
        "safety": profile.get("safety") == _safety_fields(),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"#238 frozen profile drifted: {failed}")
    lineage = dict(profile.get("lineage") or {})
    required_lineage = {
        "source_target_free_freeze_manifest_sha256",
        "source_accepted_bet_decision_freeze_sha256",
        "source_selected_index_rows_sha256",
        "source_attempted_index_rows_sha256",
        "source_target_free_feature_rows_sha256",
        "source_five_action_rows_sha256",
        "source_v6_7_candidate_rows_sha256",
        "source_v6_7_base_selected_rows_sha256",
        "runtime_policy_profile_sha256",
        "v7_4_historical_manifest_sha256",
        "v7_4_historical_noninferiority_report_sha256",
        "v7_5_historical_manifest_sha256",
        "v7_5_historical_noninferiority_report_sha256",
    }
    if set(lineage) != required_lineage:
        raise ValueError("#238 frozen lineage field set drifted")
    for name, digest in lineage.items():
        _require_sha256(str(digest), name=name)


def run_retained_v6_7_future_noninferiority(
    config: RetainedV67FutureNoninferiorityConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Run exactly one stage of the pre-registered #238 chain."""

    inputs, profile, source = _verified_source_inputs(config)
    if config.stage == "freeze_target_free":
        return _freeze_target_free(config, inputs=inputs, profile=profile, source=source)
    target_free_path = Path(config.target_free_freeze_manifest_path).resolve()
    _verify_pin(
        target_free_path,
        str(config.expected_target_free_freeze_manifest_sha256),
        "#238 target-free freeze",
    )
    target_free = _load_json(target_free_path)
    _validate_target_free_freeze(
        target_free,
        target_free_path=target_free_path,
        profile=profile,
        profile_path=inputs["profile"],
        source_path=inputs["source_freeze"],
        source=source,
    )
    inputs["target_free_freeze"] = target_free_path
    if config.stage == "settle":
        return _settle(
            config,
            inputs=inputs,
            target_free=target_free,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    runtime_profile, settled_index = _verified_evaluation_inputs(
        config,
        inputs=inputs,
        profile=profile,
        target_free=target_free,
    )
    return _evaluate(
        config,
        inputs=inputs,
        profile=profile,
        target_free=target_free,
        runtime_profile=runtime_profile,
        settled_index=settled_index,
    )


def _verified_source_inputs(
    config: RetainedV67FutureNoninferiorityConfig,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    inputs = {
        "profile": Path(config.profile_path).resolve(),
        "source_freeze": Path(config.source_freeze_manifest_path).resolve(),
    }
    if config.expected_profile_sha256 != FROZEN_PROFILE_SHA256:
        raise ValueError("#238 profile SHA is not the frozen implementation constant")
    _verify_pin(inputs["profile"], config.expected_profile_sha256, "#238 profile")
    _verify_pin(
        inputs["source_freeze"],
        config.expected_source_freeze_manifest_sha256,
        "#238 source freeze",
    )
    profile = _load_json(inputs["profile"])
    validate_retained_v6_7_future_profile(profile)
    if (
        config.expected_source_freeze_manifest_sha256
        != profile["lineage"]["source_target_free_freeze_manifest_sha256"]
    ):
        raise ValueError("#238 source freeze does not match frozen lineage")
    source = _load_json(inputs["source_freeze"])
    _validate_source_freeze(source, profile=profile)
    return inputs, profile, source


def _validate_source_freeze(source: dict[str, Any], *, profile: dict[str, Any]) -> None:
    lineage = profile["lineage"]
    expected_descriptors = {
        "accepted_bet_decision_freeze": "source_accepted_bet_decision_freeze_sha256",
        "selected_window_rows": "source_selected_index_rows_sha256",
        "attempted_window_rows": "source_attempted_index_rows_sha256",
        "target_free_feature_rows": "source_target_free_feature_rows_sha256",
        "target_free_five_action_rows": "source_five_action_rows_sha256",
        "v6_7_candidate_rows": "source_v6_7_candidate_rows_sha256",
        "v6_7_base_selected_rows": "source_v6_7_base_selected_rows_sha256",
    }
    if (
        source.get("role") != "future_confirmatory"
        or source.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or source.get("settlement_provider_called") is not False
        or source.get("source_score_mutated") is not False
        or source.get("side_count_hard_gate_enabled") is not False
    ):
        raise ValueError("#238 source freeze target-isolation contract failed")
    for field, lineage_field in expected_descriptors.items():
        descriptor = _verified_descriptor(source[field], f"#238 source {field}")
        if descriptor["sha256"] != lineage[lineage_field]:
            raise ValueError(f"#238 source descriptor drifted: {field}")
    for field, expected in _blocked_safety_fields().items():
        if source.get(field) != expected:
            raise ValueError(f"#238 source freeze safety mismatch: {field}")


def _freeze_target_free(
    config: RetainedV67FutureNoninferiorityConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(config)
    selected_descriptor = _verified_descriptor(
        source["selected_window_rows"], "#238 selected window"
    )
    decisions_descriptor = _verified_descriptor(
        source["v6_7_base_selected_rows"], "#238 retained v6.7 decisions"
    )
    decision_descriptor = _verified_descriptor(
        source["accepted_bet_decision_freeze"], "#238 source decision freeze"
    )
    selected = _load_jsonl(Path(selected_descriptor["path"]))
    decisions = _load_jsonl(Path(decisions_descriptor["path"]))
    source_decision = _load_json(Path(decision_descriptor["path"]))
    five_action_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                source["target_free_five_action_rows"], "#238 five-action rows"
            )["path"]
        )
    )
    checks, side_distribution = _target_free_checks(
        selected,
        decisions,
        source_decision=source_decision,
        stage_started_ts=config.stage_started_ts,
        minimum_support=int(
            profile["future_pnl_gate"]["minimum_guard_accepted_unique_market_count"]
        ),
    )
    checks["complete_five_action_grid"] = _complete_five_action_grid(five_action_rows)
    blockers = [
        f"target_free_{name}_failed" for name, passed in checks.items() if not passed
    ]
    passed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": BASELINE_NAME,
        "source_v6_8_candidate_status": "rejected_target_free_support",
        "source_v6_8_failure_does_not_mutate_retained_v6_7_decisions": True,
        "selected_window_market_count": len(selected),
        "retained_decision_market_count": len(decisions),
        "retained_decision_side_distribution_diagnostic": side_distribution,
        "side_quota_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "target_free_checks": checks,
        "target_free_freeze_passed": passed,
        "target_free_blocking_reason_codes": blockers,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "future_target_access_allowed": passed,
        "historical_noninferiority_gate_passed": True,
        "model_improvement_demonstrated": False,
        "candidate_decisions_are_byte_identical_to_baseline": True,
        "bounded_paper_candidate_review_allowed": False,
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "retained_v6_7_target_free_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _target_free_markdown(report))
    manifest = {
        "schema_version": TARGET_FREE_FREEZE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(inputs["profile"]),
        "source_target_free_freeze_manifest": _descriptor(inputs["source_freeze"]),
        "source_accepted_bet_decision_freeze": decision_descriptor,
        "selected_window_rows": selected_descriptor,
        "attempted_window_rows": _verified_descriptor(
            source["attempted_window_rows"], "#238 attempted window"
        ),
        "target_free_feature_rows": _verified_descriptor(
            source["target_free_feature_rows"], "#238 target-free features"
        ),
        "target_free_five_action_rows": _verified_descriptor(
            source["target_free_five_action_rows"], "#238 five-action rows"
        ),
        "v6_7_candidate_rows": _verified_descriptor(
            source["v6_7_candidate_rows"], "#238 v6.7 candidates"
        ),
        "retained_v6_7_decisions": decisions_descriptor,
        "baseline_v6_7_decisions": decisions_descriptor,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "target_free_freeze_passed": passed,
        "future_target_access_allowed": passed,
        "candidate_decisions_are_byte_identical_to_baseline": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "side_quota_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "bounded_paper_candidate_review_allowed": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "retained_v6_7_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest)


def _target_free_checks(
    selected: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    source_decision: dict[str, Any],
    stage_started_ts: int,
    minimum_support: int,
) -> tuple[dict[str, bool], dict[str, int]]:
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    decision_ids = [str(row.get("market_id") or "") for row in decisions]
    forbidden = [
        field for row in decisions for field in _find_nonempty_fields(row, FORBIDDEN_TARGET_FIELDS)
    ]
    row_valid = all(
        str(row.get("side") or "") in SIDES
        and str(row.get("action") or "") in SBC_ACTIONS
        and row.get("microstructure_safety_passed") is True
        and row.get("source_score_mutated") is False
        and row.get("labels_outcomes_resolution_or_pnl_opened") is False
        and math.isfinite(float(row.get("v6_7_base_score")))
        and float(row.get("v6_7_base_score")) > 0.0
        and int(row.get("max_input_ts") or 0) <= int(row.get("decision_ts") or 0)
        and int(row.get("decision_ts") or 0) < int(row.get("market_close_ts") or 0)
        for row in decisions
    )
    checks = {
        "exact_120_unique_selected_markets": (
            len(selected_ids) == WINDOW_MARKET_COUNT
            and "" not in selected_ids
            and len(set(selected_ids)) == WINDOW_MARKET_COUNT
        ),
        "exact_one_retained_decision_per_market": (
            len(decision_ids) == WINDOW_MARKET_COUNT
            and len(set(decision_ids)) == WINDOW_MARKET_COUNT
            and set(decision_ids) == set(selected_ids)
        ),
        "minimum_guard_accepted_market_support": len(decisions) >= minimum_support,
        "all_markets_closed_before_target_access": (
            bool(selected)
            and stage_started_ts > max(int(row.get("market_end_ts") or 0) for row in selected)
            and source_decision.get("all_selected_markets_closed_before_freeze") is True
        ),
        "decision_time_feature_causality": row_valid,
        "target_fields_absent": not forbidden,
        "source_decision_targets_sealed": (
            source_decision.get("labels_outcomes_resolution_or_pnl_opened") is False
            and source_decision.get("settlement_provider_called") is False
            and source_decision.get("source_score_mutated") is False
        ),
        "candidate_and_baseline_decisions_identical": True,
    }
    return checks, dict(sorted(Counter(str(row["side"]) for row in decisions).items()))


def _complete_five_action_grid(rows: list[dict[str, Any]]) -> bool:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        market_id = str(row.get("market_id") or "")
        decision_ts = int(row.get("decision_ts") or 0)
        action = str(row.get("action") or "")
        if (
            not market_id
            or decision_ts <= 0
            or action not in FIVE_ACTIONS
            or int(row.get("max_input_ts") or 0) > decision_ts
            or _find_nonempty_fields(row, FORBIDDEN_TARGET_FIELDS)
        ):
            return False
        groups[(market_id, decision_ts)].add(action)
    return (
        bool(groups)
        and len({market_id for market_id, _ in groups}) == WINDOW_MARKET_COUNT
        and all(actions == FIVE_ACTIONS for actions in groups.values())
    )


def _validate_target_free_freeze(
    freeze: dict[str, Any],
    *,
    target_free_path: Path,
    profile: dict[str, Any],
    profile_path: Path,
    source_path: Path,
    source: dict[str, Any],
) -> None:
    del target_free_path
    if (
        freeze.get("schema_version") != TARGET_FREE_FREEZE_SCHEMA_VERSION
        or freeze.get("profile") != _descriptor(profile_path)
        or freeze.get("source_target_free_freeze_manifest") != _descriptor(source_path)
        or freeze.get("target_free_freeze_passed") is not True
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("candidate_decisions_are_byte_identical_to_baseline") is not True
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_score_mutated") is not False
        or freeze.get("side_quota_enabled") is not False
        or freeze.get("side_pnl_hard_gate_enabled") is not False
    ):
        raise ValueError("#238 target-free freeze is not target-access eligible")
    if freeze.get("retained_v6_7_decisions") != freeze.get("baseline_v6_7_decisions"):
        raise ValueError("#238 retained and baseline decision descriptors differ")
    for field, expected in _safety_fields().items():
        if freeze.get(field) != expected:
            raise ValueError(f"#238 target-free safety mismatch: {field}")
    expected = {
        "selected_window_rows": "selected_window_rows",
        "attempted_window_rows": "attempted_window_rows",
        "target_free_feature_rows": "target_free_feature_rows",
        "target_free_five_action_rows": "target_free_five_action_rows",
        "v6_7_candidate_rows": "v6_7_candidate_rows",
        "retained_v6_7_decisions": "v6_7_base_selected_rows",
    }
    for frozen_field, source_field in expected.items():
        descriptor = _verified_descriptor(freeze[frozen_field], f"#238 {frozen_field}")
        if descriptor != _verified_descriptor(source[source_field], f"#238 source {source_field}"):
            raise ValueError(f"#238 target-free lineage mismatch: {frozen_field}")
    validate_retained_v6_7_future_profile(profile)


def _settle(
    config: RetainedV67FutureNoninferiorityConfig,
    *,
    inputs: dict[str, Path],
    target_free: dict[str, Any],
    provider_factory: Callable[[], Any] | None,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    clock_ms_fn: Callable[[], int],
) -> dict[str, Any]:
    from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider

    selected = _load_jsonl(
        Path(_verified_descriptor(target_free["selected_window_rows"], "#238 selected")['path'])
    )
    source_decision_descriptor = _verified_descriptor(
        target_free["source_accepted_bet_decision_freeze"], "#238 source decision freeze"
    )
    source_decision = _load_json(Path(source_decision_descriptor["path"]))
    if config.stage_started_ts <= int(source_decision["decision_freeze_created_ts"]):
        raise ValueError("#238 settlement attempted before decision freeze")
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected):
        raise ValueError("#238 settlement attempted before all markets closed")
    claim_path = inputs["target_free_freeze"].parent / "retained_v6_7_settlement_single_use_claim.json"
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-single-use-claim-v1",
        "run_id": config.run_id,
        "target_access_started_ts": config.stage_started_ts,
        "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
        "result_selected_rerun_allowed": False,
        **_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    _write_single_use_claim(claim_path, claim)
    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_path = Path(
        _verified_descriptor(target_free["target_free_feature_rows"], "#238 features")["path"]
    )
    for row in _load_jsonl(feature_path):
        frozen_features[str(row["market_id"])].append(row)
    run_dir = _prepare_run_dir(config)
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "target_access_started_ts": config.stage_started_ts,
        "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_enabled": False,
        **_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "retained_v6_7_settlement_started.json"
    _write_json(marker_path, marker)
    factory = provider_factory or (
        lambda: PolymarketPublicHTTPRealCorpusProvider(
            max_markets=1,
            timeout_seconds=config.provider_timeout_seconds,
            http_timeout_seconds=config.provider_http_timeout_seconds,
            use_rest_orderbooks=False,
        )
    )
    selected_by_market = {str(row["market_id"]): row for row in selected}
    pending = list(selected)
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    retried: set[str] = set()
    attempt = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending:
        attempt += 1
        results = _finalize_selected_rounds(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt,
            evaluation_only_frozen_features_by_market=frozen_features,
        )
        for result in results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"]:
                successes[market_id] = result["index_entry"]
                failures.pop(market_id, None)
            else:
                failures[market_id] = result["failure"]
        retry_ids = {
            market_id
            for market_id, failure in failures.items()
            if market_id not in successes and _is_retryable_settlement_failure(failure)
        }
        if not retry_ids:
            break
        retried.update(retry_ids)
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            for market_id in retry_ids:
                failures[market_id]["reason_codes"] = sorted(
                    {*failures[market_id].get("reason_codes", []), "settlement_max_wait_elapsed"}
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [selected_by_market[market_id] for market_id in sorted(retry_ids)]
    entries = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    unresolved = sorted(
        (failure for key, failure in failures.items() if key not in successes),
        key=lambda row: str(row["market_id"]),
    )
    complete = len(entries) == WINDOW_MARKET_COUNT and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.stage_started_ts:
        raise ValueError("#238 settlement finalization precedes target access")
    index_path = run_dir / "retained_v6_7_settled_corpus_index.json"
    if complete:
        payload = {
            "schema_version": SETTLED_INDEX_SCHEMA_VERSION,
            "run_id": config.run_id,
            "role": "future_confirmatory",
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
            "source_accepted_bet_decision_freeze_sha256": source_decision_descriptor["sha256"],
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "source_outcome_blind_rounds_mutated": False,
            "side_quota_enabled": False,
            **_safety_fields(),
        }
        payload["settled_corpus_index_id"] = canonical_json_sha256(payload)
        _write_json(index_path, payload)
    reasons = Counter(
        str(reason) for failure in unresolved for reason in failure.get("reason_codes", [])
    )
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": WINDOW_MARKET_COUNT,
        "settled_corpus_ready_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "outcomes_used_for_decision_selection_or_tuning": False,
        "source_outcome_blind_rounds_mutated": False,
        "bounded_paper_candidate_review_allowed": False,
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "retained_v6_7_settlement_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(inputs["profile"]),
        "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
        "settlement_single_use_claim": _descriptor(claim_path),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "settled_corpus_index_ready": complete,
        "bounded_paper_candidate_review_allowed": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "retained_v6_7_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        report_path,
        report,
        manifest_path,
        manifest,
        index_path if complete else None,
    )


def _verified_evaluation_inputs(
    config: RetainedV67FutureNoninferiorityConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    target_free: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_path = Path(config.runtime_policy_profile_path).resolve()
    index_path = Path(config.settled_corpus_index_path).resolve()
    if (
        config.expected_runtime_policy_profile_sha256 != FROZEN_RUNTIME_POLICY_PROFILE_SHA256
        or config.expected_runtime_policy_profile_sha256
        != profile["lineage"]["runtime_policy_profile_sha256"]
    ):
        raise ValueError("#238 runtime policy contract is not frozen")
    _verify_pin(runtime_path, str(config.expected_runtime_policy_profile_sha256), "#238 runtime")
    _verify_pin(index_path, str(config.expected_settled_corpus_index_sha256), "#238 settled index")
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if runtime_policy_source_hashes() != runtime_profile["runtime_policy_contract"][
        "source_function_sha256"
    ]:
        raise ValueError("#238 runtime policy source hashes drifted")
    settled_index = _load_json(index_path)
    _validate_settled_index(
        settled_index,
        target_free=target_free,
        target_free_path=inputs["target_free_freeze"],
        evaluation_started_ts=config.stage_started_ts,
    )
    inputs["runtime_profile"] = runtime_path
    inputs["settled_index"] = index_path
    return runtime_profile, settled_index


def _validate_settled_index(
    index: dict[str, Any],
    *,
    target_free: dict[str, Any],
    target_free_path: Path,
    evaluation_started_ts: int,
) -> None:
    selected = _load_jsonl(
        Path(_verified_descriptor(target_free["selected_window_rows"], "#238 selected")["path"])
    )
    selected_ids = {str(row["market_id"]) for row in selected}
    entries = list(index.get("entries") or [])
    entry_ids = {str(row.get("market_id") or "") for row in entries}
    if (
        index.get("schema_version") != SETTLED_INDEX_SCHEMA_VERSION
        or index.get("role") != "future_confirmatory"
        or index.get("target_free_freeze_manifest") != _descriptor(target_free_path)
        or int(index.get("entry_count") or 0) != WINDOW_MARKET_COUNT
        or len(entries) != WINDOW_MARKET_COUNT
        or entry_ids != selected_ids
        or "" in entry_ids
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
        or index.get("source_outcome_blind_rounds_mutated") is not False
    ):
        raise ValueError("#238 settled corpus index is not eligible")
    for field, expected in _safety_fields().items():
        if index.get(field) != expected:
            raise ValueError(f"#238 settled-index safety mismatch: {field}")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#238 settled entry violates quarantine contract")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"#238 settled {name}")


def _evaluate(
    config: RetainedV67FutureNoninferiorityConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    target_free: dict[str, Any],
    runtime_profile: dict[str, Any],
    settled_index: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(config)
    claim_path = inputs["target_free_freeze"].parent / (
        "retained_v6_7_future_pnl_single_use_claim.json"
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-future-pnl-single-use-claim-v1",
        "run_id": config.run_id,
        "target_evaluation_started_ts": config.stage_started_ts,
        "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "result_selected_rerun_allowed": False,
        **_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    decisions = _load_jsonl(
        Path(
            _verified_descriptor(
                target_free["retained_v6_7_decisions"], "#238 retained decisions"
            )["path"]
        )
    )
    candidate_targets = _runtime_targets_for_decisions(
        decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-candidate",
        role="future_confirmatory",
    )
    baseline_targets = _runtime_targets_for_decisions(
        decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-baseline",
        role="future_confirmatory",
    )
    selected = _load_jsonl(
        Path(_verified_descriptor(target_free["selected_window_rows"], "#238 selected")["path"])
    )
    gate = build_retained_v6_7_future_noninferiority_gate(
        candidate_targets,
        baseline_rows=baseline_targets,
        evaluation_market_ids=[str(row["market_id"]) for row in selected],
        profile=profile,
        target_free_freeze_sha256=_sha256_file(inputs["target_free_freeze"]),
    )
    candidate_path = run_dir / "retained_v6_7_candidate_runtime_targets.jsonl"
    baseline_path = run_dir / "v6_7_baseline_runtime_targets.jsonl"
    report_path = run_dir / "retained_v6_7_future_noninferiority_pnl_gate_report.json"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(baseline_path, baseline_targets)
    _write_json(report_path, gate)
    _write_text(report_path.with_suffix(".md"), _evaluation_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-future-pnl-evaluation-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "profile": _descriptor(inputs["profile"]),
        "target_free_freeze_manifest": _descriptor(inputs["target_free_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "runtime_policy_profile": _descriptor(inputs["runtime_profile"]),
        "single_use_claim": _descriptor(claim_path),
        "candidate_runtime_targets": _descriptor(candidate_path),
        "v6_7_baseline_runtime_targets": _descriptor(baseline_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "future_pnl_gate_passed": gate["future_pnl_gate_passed"],
        "future_pnl_gate_blocking_reason_codes": gate[
            "future_pnl_gate_blocking_reason_codes"
        ],
        "historical_noninferiority_gate_passed": True,
        "model_improvement_demonstrated": False,
        "bounded_paper_candidate_review_allowed": gate[
            "bounded_paper_candidate_review_allowed"
        ],
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "retained_v6_7_future_noninferiority_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, gate, manifest_path, manifest)


def build_retained_v6_7_future_noninferiority_gate(
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    evaluation_market_ids: list[str],
    profile: dict[str, Any],
    target_free_freeze_sha256: str,
) -> dict[str, Any]:
    """Apply strict absolute PnL gates and inclusive v6.7 non-inferiority."""

    validate_retained_v6_7_future_profile(profile)
    _require_sha256(target_free_freeze_sha256, name="target_free_freeze_sha256")
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    if len(market_ids) != WINDOW_MARKET_COUNT or "" in market_ids:
        raise ValueError("#238 evaluation market identity invalid")
    _validate_target_rows(candidate_rows, market_ids=market_ids)
    _validate_target_rows(baseline_rows, market_ids=market_ids)
    candidate_by_market = dict.fromkeys(market_ids, 0.0)
    baseline_by_market = dict.fromkeys(market_ids, 0.0)
    for row in candidate_rows:
        candidate_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    for row in baseline_rows:
        baseline_by_market[str(row["market_id"])] += float(
            row["runtime_policy_after_cost_net_pnl_at_frozen_size"]
        )
    delta_by_market = {
        market_id: candidate_by_market[market_id] - baseline_by_market[market_id]
        for market_id in market_ids
    }
    gate_config = profile["future_pnl_gate"]
    bootstrap = _market_bootstrap_interval(
        list(delta_by_market.values()),
        resample_count=int(gate_config["bootstrap_resample_count"]),
        confidence_level=float(gate_config["bootstrap_confidence_level"]),
        seed=int(gate_config["bootstrap_seed"]),
    )
    candidate_total = float(sum(candidate_by_market.values()))
    baseline_total = float(sum(baseline_by_market.values()))
    candidate_largest_winner = max(candidate_by_market.values(), default=0.0)
    baseline_largest_winner = max(baseline_by_market.values(), default=0.0)
    candidate_lwr = candidate_total - max(candidate_largest_winner, 0.0)
    baseline_lwr = baseline_total - max(baseline_largest_winner, 0.0)
    total_delta = candidate_total - baseline_total
    lwr_delta = candidate_lwr - baseline_lwr
    by_side: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_side[str(row["side"])].append(row)
        by_action[str(row["action"])].append(row)
    target_isolation = all(
        row.get("target_available_only_post_exit_or_official_resolution") is True
        and row.get("target_used_as_decision_time_input") is False
        and int(row["max_input_ts"]) <= int(row["decision_ts"])
        for row in candidate_rows + baseline_rows
    )
    candidate_identity = {
        str(row["market_id"]): (str(row["side"]), str(row["action"]))
        for row in candidate_rows
    }
    baseline_identity = {
        str(row["market_id"]): (str(row["side"]), str(row["action"]))
        for row in baseline_rows
    }
    exact_policy_match = (
        candidate_identity == baseline_identity
        and all(abs(value) <= 1e-12 for value in delta_by_market.values())
    )
    checks = {
        "minimum_guard_accepted_unique_market_support": len(candidate_rows)
        >= int(gate_config["minimum_guard_accepted_unique_market_count"]),
        "accepted_total_after_cost_pnl_positive": candidate_total
        > float(gate_config["accepted_total_after_cost_pnl_minimum_exclusive"]),
        "largest_winner_removed_after_cost_pnl_positive": candidate_lwr
        > float(gate_config["largest_winner_removed_after_cost_pnl_minimum_exclusive"]),
        "candidate_noninferior_to_v6_7_total_pnl": total_delta
        >= float(gate_config["candidate_minus_v6_7_after_cost_pnl_minimum_inclusive"]),
        "candidate_noninferior_to_v6_7_largest_winner_removed": lwr_delta
        >= float(
            gate_config["candidate_minus_v6_7_largest_winner_removed_minimum_inclusive"]
        ),
        "candidate_noninferior_to_v6_7_bootstrap_lcb": float(
            bootstrap["lower_confidence_bound"]
        )
        >= float(gate_config["candidate_minus_v6_7_bootstrap_lcb_minimum_inclusive"]),
        "candidate_and_v6_7_runtime_policies_identical": exact_policy_match,
        "settlement_causality_and_target_isolation": target_isolation,
    }
    reason_map = {
        "minimum_guard_accepted_unique_market_support": (
            "insufficient_guard_accepted_unique_market_support"
        ),
        "accepted_total_after_cost_pnl_positive": "accepted_total_after_cost_pnl_not_positive",
        "largest_winner_removed_after_cost_pnl_positive": (
            "largest_winner_removed_after_cost_pnl_not_positive"
        ),
        "candidate_noninferior_to_v6_7_total_pnl": "candidate_total_pnl_inferior_to_v6_7",
        "candidate_noninferior_to_v6_7_largest_winner_removed": (
            "candidate_largest_winner_removed_pnl_inferior_to_v6_7"
        ),
        "candidate_noninferior_to_v6_7_bootstrap_lcb": (
            "candidate_minus_v6_7_bootstrap_lcb_negative"
        ),
        "candidate_and_v6_7_runtime_policies_identical": (
            "retained_candidate_runtime_policy_differs_from_v6_7"
        ),
        "settlement_causality_and_target_isolation": (
            "settlement_causality_or_target_isolation_failed"
        ),
    }
    blockers = [reason_map[name] for name, passed in checks.items() if not passed]
    passed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-future-pnl-gate-report-v1",
        "candidate_name": CANDIDATE_NAME,
        "baseline_name": BASELINE_NAME,
        "target_free_freeze_sha256": target_free_freeze_sha256,
        "evaluation_market_count": len(market_ids),
        "accepted_unique_market_count": len(candidate_rows),
        "accepted_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in candidate_rows).items())
        ),
        "accepted_side_metrics_diagnostic": {
            side: _group_metrics(rows) for side, rows in sorted(by_side.items())
        },
        "accepted_action_metrics_diagnostic": {
            action: _group_metrics(rows) for action, rows in sorted(by_action.items())
        },
        "candidate_after_cost_pnl": candidate_total,
        "v6_7_after_cost_pnl": baseline_total,
        "candidate_minus_v6_7_after_cost_pnl": total_delta,
        "candidate_largest_winner_after_cost_pnl": candidate_largest_winner,
        "candidate_largest_winner_removed_after_cost_pnl": candidate_lwr,
        "v6_7_largest_winner_after_cost_pnl": baseline_largest_winner,
        "v6_7_largest_winner_removed_after_cost_pnl": baseline_lwr,
        "candidate_minus_v6_7_largest_winner_removed_after_cost_pnl": lwr_delta,
        "candidate_minus_v6_7_market_bootstrap": bootstrap,
        "noninferiority_comparison_operator": "greater_than_or_equal",
        "equality_passes_noninferiority": True,
        "historical_noninferiority_gate_passed": True,
        "future_noninferiority_gate_passed": (
            checks["candidate_noninferior_to_v6_7_total_pnl"]
            and checks["candidate_noninferior_to_v6_7_largest_winner_removed"]
            and checks["candidate_noninferior_to_v6_7_bootstrap_lcb"]
        ),
        "model_improvement_demonstrated": total_delta > 0.0 and lwr_delta > 0.0,
        "side_action_and_family_metrics_diagnostic_only": True,
        "side_quota_enabled": False,
        "side_pnl_hard_gate_enabled": False,
        "future_pnl_gate_checks": checks,
        "future_pnl_gate_passed": passed,
        "future_pnl_gate_blocking_reason_codes": blockers,
        "bounded_paper_candidate_review_allowed": passed,
        "paper_candidate_auto_unlock_allowed": False,
        "future_outcomes_used_for_model_threshold_cost_sizing_or_guard_tuning": False,
        "single_use_future_gate": True,
        "result_selected_rerun_allowed": False,
        **_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _validate_target_rows(rows: list[dict[str, Any]], *, market_ids: list[str]) -> None:
    allowed = set(market_ids)
    seen: set[str] = set()
    for row in rows:
        market_id = str(row.get("market_id") or "")
        side = str(row.get("side") or row.get("selected_side") or "")
        action = str(row.get("action") or row.get("executed_action") or "")
        if (
            not market_id
            or market_id not in allowed
            or market_id in seen
            or side not in SIDES
            or action not in SBC_ACTIONS
        ):
            raise ValueError("#238 runtime target identity invalid")
        seen.add(market_id)
        value = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        if not math.isfinite(value):
            raise ValueError("#238 runtime target PnL is non-finite")


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"]) for row in rows]
    return {
        "accepted_bet_count": len(rows),
        "accepted_unique_market_count": len({str(row["market_id"]) for row in rows}),
        "after_cost_pnl_sum": float(sum(values)),
        "after_cost_pnl_mean": float(sum(values) / len(values)) if values else 0.0,
        "win_rate": float(sum(value > 0.0 for value in values) / len(values)) if values else 0.0,
        "diagnostic_only": True,
    }


def _prepare_run_dir(config: RetainedV67FutureNoninferiorityConfig) -> Path:
    run_dir = Path(config.output_dir).resolve() / config.run_id
    if run_dir.exists():
        if not config.overwrite_existing:
            raise FileExistsError(f"run path exists: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def _result(
    run_dir: Path,
    report_path: Path,
    report: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    index_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "run_dir": run_dir,
        "report": report,
        "report_path": report_path,
        "report_sha256": _sha256_file(report_path),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "index_path": index_path,
        "index_sha256": _sha256_file(index_path) if index_path is not None else None,
    }


def _target_free_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Retained v6.7 Exact-120 Target-Free Freeze",
            "",
            f"- freeze passed: `{str(report['target_free_freeze_passed']).lower()}`",
            f"- selected markets: `{report['selected_window_market_count']}`",
            f"- retained decisions: `{report['retained_decision_market_count']}`",
            f"- side distribution (diagnostic): `{report['retained_decision_side_distribution_diagnostic']}`",
            f"- blockers: `{report['target_free_blocking_reason_codes']}`",
            "- equality to v6.7 passes non-inferiority: `true`",
            "- outcomes/PnL opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Retained v6.7 Exact-120 Read-Only Settlement",
            "",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            f"- settled index ready: `{str(report['settled_corpus_index_ready']).lower()}`",
            "- source outcome-blind artifacts mutated: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Retained v6.7 Exact-120 Future Non-Inferiority PnL Gate",
            "",
            f"- future PnL gate passed: `{str(report['future_pnl_gate_passed']).lower()}`",
            f"- blockers: `{report['future_pnl_gate_blocking_reason_codes']}`",
            f"- candidate / v6.7 PnL: `{report['candidate_after_cost_pnl']} / {report['v6_7_after_cost_pnl']}`",
            f"- candidate-minus-v6.7 PnL: `{report['candidate_minus_v6_7_after_cost_pnl']}`",
            f"- largest-winner-removed PnL: `{report['candidate_largest_winner_removed_after_cost_pnl']}`",
            f"- future non-inferiority passed: `{str(report['future_noninferiority_gate_passed']).lower()}`",
            f"- model improvement demonstrated: `{str(report['model_improvement_demonstrated']).lower()}`",
            f"- bounded paper-candidate review allowed: `{str(report['bounded_paper_candidate_review_allowed']).lower()}`",
            "- equality passes non-inferiority; improvement is reported separately",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "FROZEN_PROFILE_SHA256",
    "FROZEN_RUNTIME_POLICY_PROFILE_SHA256",
    "RetainedV67FutureNoninferiorityConfig",
    "build_retained_v6_7_future_noninferiority_gate",
    "run_retained_v6_7_future_noninferiority",
    "validate_retained_v6_7_future_profile",
]
