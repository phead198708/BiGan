"""Exact-195 powered future gate for the retained v6.7 champion (#252)."""

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
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    materialize_adaptive_support_controller_v8_1_runtime_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    _baseline_guard_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_post_freeze import (
    _write_single_use_claim,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    build_v6_7_target_free_candidate_rows,
    select_v6_7_target_free_rows,
    validate_p_up_semantic_compatibility_v6_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    _verify_index_raw_descriptors,
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _descriptor,
    _find_nonempty_fields,
    _load_json,
    _load_jsonl,
    _require_sha256,
    _sha256_file,
    _verified_descriptor,
    _verify_pin,
    _write_json,
    _write_jsonl,
    _write_text,
)
from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_paper_readiness import (
    GATE_PLAN_SCHEMA_VERSION,
    ISSUE238_MANIFEST_SCHEMA,
    ISSUE250_MANIFEST_SCHEMA,
    SAFETY,
    _market_bootstrap_mean,
    validate_retained_v6_7_paper_readiness_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_retained_v6_7_paper_readiness import (
    MANIFEST_SCHEMA_VERSION as READINESS_MANIFEST_SCHEMA_VERSION,
)
from bigan.v8.polymarket.training.execution_layer_v2_runtime_aligned_sbc_net_return_v6_4 import (
    runtime_policy_source_hashes,
    validate_runtime_aligned_sbc_net_return_v6_4_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

SCHEMA_PREFIX = "bigan-v8-retained-v6-7-powered-paper-gate"
TARGET_FREE_FREEZE_SCHEMA_VERSION = (
    f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1"
)
SETTLED_INDEX_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-settled-corpus-index-v1"
STAGES = {"freeze_target_free", "settle", "evaluate_powered_pnl"}
EXACT_MARKET_COUNT = 195
MAXIMUM_CAPTURE_ATTEMPT_COUNT = 228
MINIMUM_GUARD_ACCEPTED_MARKET_COUNT = 186
FROZEN_V6_7_PROFILE_SHA256 = (
    "cec55d243acd6bbf60a5e8474545b487086ddcd4d18073682ae7f2d4660d2248"
)
FROZEN_RUNTIME_POLICY_PROFILE_SHA256 = (
    "1306f6b6f7a6c1216b23413352ff66f4061ec62a9751b0de51eded256ca51264"
)
FROZEN_FEATURE_CONTRACT_SHA256 = (
    "a4819ad6beec8d72612aa25ef2af751c357e807d514dcf1d2c94b37eba07c959"
)
FROZEN_V6_2_CANDIDATE_MANIFEST_SHA256 = (
    "b9441b04fb595a927cbf9af9311612b037c36fc8c623ac8a92b6f4cb8ece84b9"
)
FIVE_ACTIONS = frozenset(
    {
        "BUY_UP_SELL_BEFORE_CLOSE",
        "BUY_DOWN_SELL_BEFORE_CLOSE",
        "BUY_UP_HOLD_TO_SETTLEMENT",
        "BUY_DOWN_HOLD_TO_SETTLEMENT",
        "NO_TRADE",
    }
)
FORBIDDEN_TARGET_FIELDS = frozenset(FORBIDDEN_INFERENCE_FIELDS) | frozenset(
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


@dataclass(frozen=True, slots=True)
class RetainedV67PoweredPaperGateConfig:
    """Pinned inputs for one #252 freeze, settlement, or PnL stage."""

    stage: Literal["freeze_target_free", "settle", "evaluate_powered_pnl"]
    run_id: str
    output_dir: Path | str
    readiness_manifest_path: Path | str
    expected_readiness_manifest_sha256: str
    implementation_commit: str
    stage_started_ts: int
    collector_protocol_path: Path | str | None = None
    expected_collector_protocol_sha256: str | None = None
    collector_index_path: Path | str | None = None
    expected_collector_index_sha256: str | None = None
    v6_7_profile_path: Path | str | None = None
    expected_v6_7_profile_sha256: str | None = None
    development_batch_manifest_paths: tuple[Path | str, ...] = ()
    expected_development_batch_manifest_sha256s: tuple[str, ...] = ()
    v6_2_batch_manifest_paths: tuple[Path | str, ...] = ()
    expected_v6_2_batch_manifest_sha256s: tuple[str, ...] = ()
    target_free_freeze_manifest_path: Path | str | None = None
    expected_target_free_freeze_manifest_sha256: str | None = None
    settled_corpus_index_path: Path | str | None = None
    expected_settled_corpus_index_sha256: str | None = None
    runtime_policy_profile_path: Path | str | None = None
    expected_runtime_policy_profile_sha256: str | None = None
    provider_timeout_seconds: float = 15.0
    provider_http_timeout_seconds: float = 5.0
    settlement_max_wait_seconds: float = 600.0
    settlement_poll_interval_seconds: float = 15.0
    max_workers: int = 8
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if (
            self.stage not in STAGES
            or not self.run_id.strip()
            or len(self.implementation_commit) != 40
            or self.stage_started_ts <= 0
        ):
            raise ValueError("#252 stage configuration is invalid")
        _require_sha256(
            self.expected_readiness_manifest_sha256,
            name="expected_readiness_manifest_sha256",
        )
        required_by_stage = {
            "freeze_target_free": (
                "collector_protocol_path",
                "expected_collector_protocol_sha256",
                "collector_index_path",
                "expected_collector_index_sha256",
                "v6_7_profile_path",
                "expected_v6_7_profile_sha256",
            ),
            "settle": (
                "target_free_freeze_manifest_path",
                "expected_target_free_freeze_manifest_sha256",
            ),
            "evaluate_powered_pnl": (
                "target_free_freeze_manifest_path",
                "expected_target_free_freeze_manifest_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
            ),
        }
        for name in required_by_stage[self.stage]:
            if getattr(self, name) in (None, ""):
                raise ValueError(f"#252 required stage input missing: {name}")
        for name in (
            "expected_collector_protocol_sha256",
            "expected_collector_index_sha256",
            "expected_v6_7_profile_sha256",
            "expected_target_free_freeze_manifest_sha256",
            "expected_settled_corpus_index_sha256",
            "expected_runtime_policy_profile_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_sha256(str(value), name=name)
        if self.stage == "freeze_target_free":
            if (
                not self.development_batch_manifest_paths
                or len(self.development_batch_manifest_paths)
                != len(self.expected_development_batch_manifest_sha256s)
                or len(self.v6_2_batch_manifest_paths)
                != len(self.expected_v6_2_batch_manifest_sha256s)
                or len(self.development_batch_manifest_paths)
                != len(self.v6_2_batch_manifest_paths)
            ):
                raise ValueError("#252 aligned sealed batch manifest pins are required")
            for digest in (
                *self.expected_development_batch_manifest_sha256s,
                *self.expected_v6_2_batch_manifest_sha256s,
            ):
                _require_sha256(digest, name="batch_manifest_sha256")
        if (
            self.provider_timeout_seconds <= 0
            or self.provider_http_timeout_seconds <= 0
            or self.settlement_max_wait_seconds < 0
            or self.settlement_poll_interval_seconds <= 0
            or self.max_workers <= 0
        ):
            raise ValueError("#252 settlement bounds are invalid")
        for name in (
            "output_dir",
            "readiness_manifest_path",
            "collector_protocol_path",
            "collector_index_path",
            "v6_7_profile_path",
            "target_free_freeze_manifest_path",
            "settled_corpus_index_path",
            "runtime_policy_profile_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(
            self,
            "development_batch_manifest_paths",
            tuple(Path(path) for path in self.development_batch_manifest_paths),
        )
        object.__setattr__(
            self,
            "v6_2_batch_manifest_paths",
            tuple(Path(path) for path in self.v6_2_batch_manifest_paths),
        )


def run_retained_v6_7_powered_paper_gate(
    config: RetainedV67PoweredPaperGateConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Run one strictly separated stage of the #252 powered gate."""

    readiness_path, readiness, profile, gate_plan = _verified_readiness(config)
    if config.stage == "freeze_target_free":
        return _freeze_target_free(
            config,
            readiness_path=readiness_path,
            readiness=readiness,
            profile=profile,
            gate_plan=gate_plan,
        )
    freeze_path = Path(config.target_free_freeze_manifest_path).resolve()
    freeze, selected, decisions = _validated_target_free_freeze(
        freeze_path,
        expected_sha256=str(config.expected_target_free_freeze_manifest_sha256),
        readiness_path=readiness_path,
    )
    if config.stage == "settle":
        return _settle(
            config,
            readiness_path=readiness_path,
            freeze_path=freeze_path,
            freeze=freeze,
            selected=selected,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    return _evaluate(
        config,
        readiness_path=readiness_path,
        freeze_path=freeze_path,
        freeze=freeze,
        selected=selected,
        decisions=decisions,
        gate_plan=gate_plan,
    )


def validate_powered_gate_plan(gate_plan: dict[str, Any]) -> None:
    """Reject drift from the powered gate frozen before collection."""

    checks = {
        "schema": gate_plan.get("schema_version") == GATE_PLAN_SCHEMA_VERSION,
        "frozen": gate_plan.get("frozen") is True,
        "champion": gate_plan.get("champion") == "retained_v6_7_champion",
        "exact_count": gate_plan.get("exact_quality_valid_market_count")
        == EXACT_MARKET_COUNT,
        "attempt_cap": gate_plan.get("maximum_capture_attempt_count")
        == MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "support": gate_plan.get("minimum_target_free_guard_accepted_market_count")
        == MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "batch_size": gate_plan.get("bounded_batch_market_count") == 12,
        "identity_fields": gate_plan.get("required_disjoint_identity_fields")
        == ["market_id", "slug", "decision_id", "source_row_hash"],
        "target_free": gate_plan.get("target_free_decision_freeze_required") is True,
        "settlement": gate_plan.get(
            "official_read_only_settlement_on_quarantine_copies"
        )
        is True
        and gate_plan.get("complete_settlement_required") is True,
        "single_use": gate_plan.get("one_evaluation_only") is True
        and gate_plan.get("result_selected_rerun_allowed") is False
        and gate_plan.get("result_selected_extension_allowed") is False,
        "no_side_quota": gate_plan.get("side_quota_enabled") is False,
        "manual_unlock": gate_plan.get("paper_candidate_auto_unlock_allowed") is False
        and gate_plan.get("separate_manual_paper_authorization_issue_required")
        is True,
        "safety": all(
            gate_plan.get(field) == expected for field, expected in SAFETY.items()
        ),
    }
    hard_gate = dict(gate_plan.get("hard_gate_checks") or {})
    checks["hard_gate"] = hard_gate == {
        "total_after_cost_pnl_minimum_exclusive": 0.0,
        "largest_winner_removed_after_cost_pnl_minimum_exclusive": 0.0,
        "market_bootstrap_one_sided_lower_bound_minimum_exclusive": 0.0,
        "market_bootstrap_seed": 2522026,
        "market_bootstrap_resample_count": 10000,
        "market_bootstrap_one_sided_confidence_level": 0.95,
        "runtime_safety_and_forbidden_field_checks_required": True,
    }
    blockers = sorted(name for name, passed in checks.items() if not passed)
    if blockers:
        raise ValueError("#252 powered gate plan drifted: " + ", ".join(blockers))


def select_powered_target_free_window(
    index_rows: list[dict[str, Any]],
    *,
    gate_plan: dict[str, Any],
    prior_registries: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select the earliest exact-195 eligible markets within the frozen cap."""

    validate_powered_gate_plan(gate_plan)
    ordered = sorted(index_rows, key=lambda row: int(row.get("sequence") or 0))
    attempted: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    selected_values = {field: set() for field in prior_registries}
    contract_violations: Counter[str] = Counter()
    boundary = int(gate_plan["strictly_later_minimum_market_start_ts_exclusive"])
    for row in ordered[:MAXIMUM_CAPTURE_ATTEMPT_COUNT]:
        attempted.append(row)
        if row.get("capture_quality_valid") is not True:
            exclusions["capture_quality_invalid"] += 1
            continue
        if int(row.get("market_start_ts") or 0) <= boundary:
            exclusions["market_start_not_strictly_later"] += 1
            contract_violations["market_start_not_strictly_later"] += 1
            continue
        row_values = {
            "market_id": str(row.get("market_id") or ""),
            "slug": str(row.get("slug") or ""),
            "decision_id": str(row.get("decision_id") or ""),
            "source_row_hash": str(row.get("source_row_hash") or ""),
        }
        if any(not value for value in row_values.values()):
            exclusions["required_identity_missing"] += 1
            contract_violations["required_identity_missing"] += 1
            continue
        overlap_fields = [
            field
            for field, value in row_values.items()
            if value in prior_registries[field]
        ]
        duplicate_fields = [
            field
            for field, value in row_values.items()
            if value in selected_values[field]
        ]
        if overlap_fields:
            for field in overlap_fields:
                exclusions[f"prior_{field}_overlap"] += 1
                contract_violations[f"prior_{field}_overlap"] += 1
            continue
        if duplicate_fields:
            for field in duplicate_fields:
                exclusions[f"selected_{field}_duplicate"] += 1
                contract_violations[f"selected_{field}_duplicate"] += 1
            continue
        selected.append(row)
        for field, value in row_values.items():
            selected_values[field].add(value)
        if len(selected) == EXACT_MARKET_COUNT:
            break
    summary = {
        "attempted_scan_count": len(attempted),
        "selected_market_count": len(selected),
        "exact_window_ready": len(selected) == EXACT_MARKET_COUNT,
        "attempt_cap_respected": len(attempted) <= MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "selected_sequence_start": (
            int(selected[0]["sequence"]) if selected else None
        ),
        "selected_sequence_end": (
            int(selected[-1]["sequence"]) if selected else None
        ),
        "exclusion_reason_distribution": dict(sorted(exclusions.items())),
        "collection_contract_violation_distribution": dict(
            sorted(contract_violations.items())
        ),
        "collection_contract_violation_count": sum(contract_violations.values()),
        "selected_market_ids_sha256": canonical_json_sha256(
            [str(row["market_id"]) for row in selected]
        ),
    }
    return selected, attempted, summary


def build_powered_pnl_gate(
    target_rows: list[dict[str, Any]],
    *,
    evaluation_market_ids: list[str],
    minimum_guard_accepted_market_count: int,
    gate_plan: dict[str, Any],
    target_free_freeze_sha256: str,
) -> dict[str, Any]:
    """Apply the preregistered absolute PnL, LWR, and powered LCB gates."""

    validate_powered_gate_plan(gate_plan)
    _require_sha256(
        target_free_freeze_sha256, name="target_free_freeze_sha256"
    )
    market_ids = list(dict.fromkeys(str(value) for value in evaluation_market_ids))
    if len(market_ids) != EXACT_MARKET_COUNT or "" in market_ids:
        raise ValueError("#252 evaluation market identity invalid")
    allowed = set(market_ids)
    target_by_market: dict[str, dict[str, Any]] = {}
    for row in target_rows:
        market_id = str(row.get("market_id") or "")
        pnl = float(row.get("runtime_policy_after_cost_net_pnl_at_frozen_size"))
        if (
            market_id not in allowed
            or market_id in target_by_market
            or not math.isfinite(pnl)
            or row.get("target_available_only_post_exit_or_official_resolution")
            is not True
            or row.get("target_used_as_decision_time_input") is not False
            or int(row.get("max_input_ts") or 0)
            > int(row.get("decision_ts") or 0)
        ):
            raise ValueError("#252 runtime target row is invalid")
        target_by_market[market_id] = row
    market_pnl = {
        market_id: float(
            target_by_market.get(market_id, {}).get(
                "runtime_policy_after_cost_net_pnl_at_frozen_size", 0.0
            )
        )
        for market_id in market_ids
    }
    values = list(market_pnl.values())
    total = float(sum(values))
    largest_winner = max(values, default=0.0)
    largest_winner_removed = total - max(largest_winner, 0.0)
    hard_gate = gate_plan["hard_gate_checks"]
    bootstrap = _market_bootstrap_mean(
        values,
        seed=int(hard_gate["market_bootstrap_seed"]),
        samples=int(hard_gate["market_bootstrap_resample_count"]),
        one_sided_confidence=float(
            hard_gate["market_bootstrap_one_sided_confidence_level"]
        ),
    )
    by_side: dict[str, list[float]] = defaultdict(list)
    by_action: dict[str, list[float]] = defaultdict(list)
    for row in target_rows:
        pnl = float(row["runtime_policy_after_cost_net_pnl_at_frozen_size"])
        by_side[str(row["side"])].append(pnl)
        by_action[str(row["action"])].append(pnl)
    checks = {
        "exact_quality_valid_market_count": len(market_ids)
        == EXACT_MARKET_COUNT,
        "minimum_guard_accepted_market_support": len(target_rows)
        >= minimum_guard_accepted_market_count,
        "total_after_cost_pnl_positive": total
        > float(hard_gate["total_after_cost_pnl_minimum_exclusive"]),
        "largest_winner_removed_after_cost_pnl_positive": largest_winner_removed
        > float(
            hard_gate[
                "largest_winner_removed_after_cost_pnl_minimum_exclusive"
            ]
        ),
        "market_bootstrap_one_sided_lcb_positive": float(
            bootstrap["one_sided_lower_confidence_bound"]
        )
        > float(
            hard_gate[
                "market_bootstrap_one_sided_lower_bound_minimum_exclusive"
            ]
        ),
        "target_isolation": all(
            row.get("target_available_only_post_exit_or_official_resolution")
            is True
            and row.get("target_used_as_decision_time_input") is False
            for row in target_rows
        ),
    }
    blockers = [
        f"powered_paper_gate_{name}_failed"
        for name, passed in checks.items()
        if not passed
    ]
    passed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-gate-report-v1",
        "evaluation_market_count": len(market_ids),
        "guard_accepted_unique_market_count": len(target_rows),
        "guard_blocked_no_bet_zero_market_count": len(market_ids)
        - len(target_rows),
        "total_after_cost_pnl": total,
        "mean_after_cost_pnl_per_quality_valid_market": total / len(market_ids),
        "largest_winner_after_cost_pnl": largest_winner,
        "largest_winner_removed_after_cost_pnl": largest_winner_removed,
        "market_bootstrap": bootstrap,
        "pnl_by_side_diagnostic": {
            side: {
                "accepted_market_count": len(values),
                "after_cost_pnl": float(sum(values)),
            }
            for side, values in sorted(by_side.items())
        },
        "pnl_by_action_diagnostic": {
            action: {
                "accepted_market_count": len(values),
                "after_cost_pnl": float(sum(values)),
            }
            for action, values in sorted(by_action.items())
        },
        "side_quota_enabled": False,
        "side_action_and_family_metrics_diagnostic_only": True,
        "powered_paper_candidate_readiness_checks": checks,
        "powered_paper_candidate_readiness_gate_passed": passed,
        "powered_paper_candidate_readiness_blocking_reason_codes": blockers,
        "target_free_freeze_sha256": target_free_freeze_sha256.lower(),
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "result_selected_extension_allowed": False,
        "manual_paper_authorization_review_eligible": passed,
        "paper_candidate_auto_unlock_allowed": False,
        "separate_manual_paper_authorization_issue_required": True,
        **SAFETY,
    }
    report["report_id"] = canonical_json_sha256(report)
    return report


def _verified_readiness(
    config: RetainedV67PoweredPaperGateConfig,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    readiness_path = config.readiness_manifest_path.resolve()
    _verify_pin(
        readiness_path,
        config.expected_readiness_manifest_sha256,
        "#252 readiness manifest",
    )
    readiness = _load_json(readiness_path)
    if (
        readiness.get("schema_version") != READINESS_MANIFEST_SCHEMA_VERSION
        or readiness.get("evidence_inventory_passed") is not True
        or readiness.get("power_analysis_ready") is not True
        or readiness.get("paper_candidate_gate_design_ready") is not True
        or readiness.get(
            "completed_future_outcomes_used_for_model_or_threshold_tuning"
        )
        is not False
        or not all(
            readiness.get(field) == expected
            for field, expected in SAFETY.items()
        )
    ):
        raise ValueError("#252 readiness manifest is not eligible")
    profile_path = Path(
        _verified_descriptor(readiness["profile"], "#252 readiness profile")[
            "path"
        ]
    )
    gate_plan_path = Path(
        _verified_descriptor(
            readiness["forward_gate_plan"], "#252 readiness gate plan"
        )["path"]
    )
    profile = _load_json(profile_path)
    gate_plan = _load_json(gate_plan_path)
    validate_retained_v6_7_paper_readiness_profile(profile)
    validate_powered_gate_plan(gate_plan)
    if (
        readiness["profile"]["sha256"]
        != _sha256_file(profile_path)
        or readiness["forward_gate_plan"]["sha256"]
        != _sha256_file(gate_plan_path)
    ):
        raise ValueError("#252 readiness descriptor drifted")
    return readiness_path, readiness, profile, gate_plan


def _prior_registries(
    readiness: dict[str, Any],
    gate_plan: dict[str, Any],
) -> dict[str, set[str]]:
    issue238_manifest = _load_json(
        Path(
            _verified_descriptor(
                readiness["issue238_manifest"], "#252 issue238 manifest"
            )["path"]
        )
    )
    issue250_manifest = _load_json(
        Path(
            _verified_descriptor(
                readiness["issue250_manifest"], "#252 issue250 manifest"
            )["path"]
        )
    )
    if (
        issue238_manifest.get("schema_version") != ISSUE238_MANIFEST_SCHEMA
        or issue250_manifest.get("schema_version") != ISSUE250_MANIFEST_SCHEMA
    ):
        raise ValueError("#252 prior target-free manifest schema invalid")
    issue238_freeze = _load_json(
        Path(
            _verified_descriptor(
                issue238_manifest["target_free_freeze_manifest"],
                "#252 issue238 target-free freeze",
            )["path"]
        )
    )
    issue250_freeze = _load_json(
        Path(
            _verified_descriptor(
                issue250_manifest["target_free_freeze_manifest"],
                "#252 issue250 target-free freeze",
            )["path"]
        )
    )
    if (
        issue238_freeze.get("labels_outcomes_resolution_or_pnl_opened")
        is not False
        or issue238_freeze.get("settlement_provider_called") is not False
        or issue250_freeze.get("labels_outcomes_resolution_or_pnl_opened")
        is not False
        or issue250_freeze.get("settlement_provider_called") is not False
    ):
        raise ValueError("#252 prior identity source is not target-free")
    prior_rows = [
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    issue238_freeze["selected_window_rows"],
                    "#252 issue238 target-free selected rows",
                )["path"]
            )
        ),
        *_load_jsonl(
            Path(
                _verified_descriptor(
                    issue250_freeze["selected_rows"],
                    "#252 issue250 target-free selected rows",
                )["path"]
            )
        ),
    ]
    forbidden = _find_nonempty_fields(prior_rows, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        raise ValueError(
            "#252 prior target-free identity rows contain targets: "
            + ",".join(sorted(forbidden))
        )
    source_keys = {
        "market_id": "market_id",
        "slug": "slug",
        "decision_id": "decision_id",
        "source_row_hash": "source_row_hash",
    }
    registries = {
        field: {str(row.get(key) or "") for row in prior_rows} - {""}
        for field, key in source_keys.items()
    }
    for field, values in registries.items():
        if (
            len(values) != gate_plan["prior_identity_registry_counts"][field]
            or canonical_json_sha256(sorted(values))
            != gate_plan["prior_identity_registry_hashes"][field]
        ):
            raise ValueError(f"#252 prior {field} registry drifted")
    return registries


def _freeze_target_free(
    config: RetainedV67PoweredPaperGateConfig,
    *,
    readiness_path: Path,
    readiness: dict[str, Any],
    profile: dict[str, Any],
    gate_plan: dict[str, Any],
) -> dict[str, Any]:
    protocol_path = Path(config.collector_protocol_path).resolve()
    index_path = Path(config.collector_index_path).resolve()
    v6_7_profile_path = Path(config.v6_7_profile_path).resolve()
    _verify_pin(
        protocol_path,
        str(config.expected_collector_protocol_sha256),
        "#252 collector protocol",
    )
    _verify_pin(
        index_path,
        str(config.expected_collector_index_sha256),
        "#252 collector index",
    )
    _verify_pin(
        v6_7_profile_path,
        str(config.expected_v6_7_profile_sha256),
        "#252 v6.7 profile",
    )
    if (
        config.expected_v6_7_profile_sha256 != FROZEN_V6_7_PROFILE_SHA256
        or config.expected_v6_7_profile_sha256
        != profile["champion_contract"]["profile_sha256"]
    ):
        raise ValueError("#252 retained v6.7 champion profile drifted")
    v6_7_profile = _load_json(v6_7_profile_path)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    prior_registries = _prior_registries(readiness, gate_plan)
    preflight_index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    _, _, preflight_selection = select_powered_target_free_window(
        preflight_index_rows,
        gate_plan=gate_plan,
        prior_registries=prior_registries,
    )
    if not preflight_selection["exact_window_ready"]:
        raise ValueError("#252 exact-195 target-free window is not ready")
    run_dir = _prepare_run_dir(
        config.output_dir.resolve(),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    index_snapshot = run_dir / "powered_v6_7_collector_index_snapshot.jsonl"
    shutil.copyfile(index_path, index_snapshot)
    if _sha256_file(index_snapshot) != str(
        config.expected_collector_index_sha256
    ).lower():
        raise ValueError("#252 collector index changed during snapshot")
    index_rows = load_and_validate_persistent_outcome_blind_index(index_snapshot)
    selected, attempted, selection = select_powered_target_free_window(
        index_rows,
        gate_plan=gate_plan,
        prior_registries=prior_registries,
    )
    if not selection["exact_window_ready"]:
        raise ValueError("#252 exact-195 target-free window is not ready")
    for row in selected:
        _verify_index_raw_descriptors(row)
    development, v6_2 = _load_sealed_batch_manifests(config)
    actions, features, scored = _load_target_free_batch_rows(
        development,
        v6_2,
        selected_market_ids={str(row["market_id"]) for row in selected},
    )
    forbidden = sorted(
        set(_find_nonempty_fields(actions, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(features, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(scored, FORBIDDEN_TARGET_FIELDS))
    )
    source_hash_before = canonical_json_sha256(
        {"actions": actions, "features": features, "scored": scored}
    )
    v6_7_candidates, candidate_summary = build_v6_7_target_free_candidate_rows(
        scored,
        action_rows=actions,
        profile=v6_7_profile,
    )
    selected_rows = select_v6_7_target_free_rows(
        v6_7_candidates,
        profile=v6_7_profile,
    )
    selected_ids = [str(row["market_id"]) for row in selected]
    guard_rows = _baseline_guard_window(
        selected_ids,
        baseline_rows=selected_rows,
        action_rows=actions,
        v6_7_profile=v6_7_profile,
    )
    runtime_rows = materialize_adaptive_support_controller_v8_1_runtime_decisions(
        guard_rows,
        action_rows=actions,
    )
    source_hash_after = canonical_json_sha256(
        {"actions": actions, "features": features, "scored": scored}
    )
    selected_id_set = set(selected_ids)
    feature_market_ids = {str(row.get("market_id") or "") for row in features}
    guard_market_ids = {str(row.get("market_id") or "") for row in guard_rows}
    runtime_market_ids = {str(row.get("market_id") or "") for row in runtime_rows}
    feature_causality_violations = sum(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in features + scored + actions
    )
    checks = {
        "exact_195_selected_markets": len(selected) == EXACT_MARKET_COUNT,
        "attempted_scan_cap_respected": len(attempted)
        <= MAXIMUM_CAPTURE_ATTEMPT_COUNT,
        "collection_contract_clean": selection[
            "collection_contract_violation_count"
        ]
        == 0,
        "all_selected_markets_closed_before_target_access": config.stage_started_ts
        > max(int(row["market_end_ts"]) for row in selected),
        "complete_five_action_grid": _complete_five_action_grid(
            actions, selected_id_set
        ),
        "complete_frozen_feature_coverage": feature_market_ids
        == selected_id_set,
        "complete_guard_decision_coverage": guard_market_ids == selected_id_set,
        "runtime_decisions_subset_selected": runtime_market_ids
        <= selected_id_set,
        "minimum_guard_accepted_market_support": len(runtime_market_ids)
        >= MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        "feature_timestamp_causality": feature_causality_violations == 0,
        "forbidden_target_fields_absent": not forbidden,
        "source_scores_unchanged": source_hash_before == source_hash_after,
        "outcomes_resolution_labels_or_pnl_sealed": all(
            row.get("labels_outcomes_or_pnl_opened") is False
            and row.get("resolution_provider_called") is False
            for row in selected
        ),
    }
    blockers = [
        f"target_free_{name}_failed"
        for name, passed in checks.items()
        if not passed
    ]
    passed = not blockers
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-report-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "decision_freeze_created_ts": config.stage_started_ts,
        "selected_market_count": len(selected),
        "attempted_market_count": len(attempted),
        "guard_accepted_market_count": len(runtime_market_ids),
        "guard_blocked_no_bet_market_count": len(selected) - len(runtime_market_ids),
        "selected_action_distribution": dict(
            sorted(
                Counter(
                    str(row.get("selected_action") or "NO_TRADE")
                    for row in guard_rows
                ).items()
            )
        ),
        "selected_side_distribution_diagnostic": dict(
            sorted(Counter(str(row["side"]) for row in runtime_rows).items())
        ),
        "selection_summary": selection,
        "v6_7_candidate_summary": candidate_summary,
        "feature_causality_violation_count": feature_causality_violations,
        "forbidden_target_fields": forbidden,
        "target_free_checks": checks,
        "target_free_freeze_passed": passed,
        "target_free_blocking_reason_codes": blockers,
        "future_target_access_allowed": passed,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_model_cost_sizing_guard_or_gate_tuning_performed": False,
        "side_quota_enabled": False,
        "side_action_and_family_metrics_diagnostic_only": True,
        "paper_candidate_auto_unlock_allowed": False,
        **SAFETY,
    }
    report["report_id"] = canonical_json_sha256(report)
    return _write_freeze_outputs(
        run_dir=run_dir,
        report=report,
        readiness_path=readiness_path,
        gate_plan_path=Path(readiness["forward_gate_plan"]["path"]),
        protocol_path=protocol_path,
        index_snapshot=index_snapshot,
        v6_7_profile_path=v6_7_profile_path,
        selected=selected,
        attempted=attempted,
        actions=actions,
        features=features,
        scored=scored,
        candidates=v6_7_candidates,
        selected_rows=selected_rows,
        guard_rows=guard_rows,
        runtime_rows=runtime_rows,
        development=development,
        v6_2=v6_2,
    )


def _load_sealed_batch_manifests(
    config: RetainedV67PoweredPaperGateConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    v6_2: list[dict[str, Any]] = []
    for dev_path, dev_pin, score_path, score_pin in zip(
        config.development_batch_manifest_paths,
        config.expected_development_batch_manifest_sha256s,
        config.v6_2_batch_manifest_paths,
        config.expected_v6_2_batch_manifest_sha256s,
        strict=True,
    ):
        dev_path = dev_path.resolve()
        score_path = score_path.resolve()
        _verify_pin(dev_path, dev_pin, "#252 development batch manifest")
        _verify_pin(score_path, score_pin, "#252 v6.2 batch manifest")
        dev = _load_json(dev_path)
        score = _load_json(score_path)
        feature_contract = _verified_descriptor(
            dev["feature_contract"], "#252 feature contract"
        )
        candidate = _verified_descriptor(
            score["candidate_manifest"], "#252 v6.2 candidate"
        )
        matched_dev = _verified_descriptor(
            score["development_batch_canary_manifest"],
            "#252 matched development batch",
        )
        score_report = _load_json(
            Path(
                _verified_descriptor(
                    score["report"], "#252 v6.2 batch report"
                )["path"]
            )
        )
        if (
            dev.get("development_data_canary_passed") is not True
            or dev.get("candidate_model_scoring_attempted") is not False
            or dev.get("labels_outcomes_or_pnl_opened") is not False
            or score.get("labels_outcomes_or_pnl_opened") is not False
            or score.get("batch_id") != dev.get("batch_id")
            or matched_dev["sha256"] != dev_pin.lower()
            or feature_contract["sha256"] != FROZEN_FEATURE_CONTRACT_SHA256
            or candidate["sha256"] != FROZEN_V6_2_CANDIDATE_MANIFEST_SHA256
            or score_report.get("target_free_scoring_passed") is not True
            or score_report.get("labels_outcomes_or_pnl_opened") is not False
            or score_report.get("settlement_provider_called") is not False
            or score_report.get("threshold_or_guard_tuning_performed") is not False
        ):
            raise ValueError("#252 sealed batch lineage is invalid")
        dev["_manifest_path"] = str(dev_path)
        score["_manifest_path"] = str(score_path)
        development.append(dev)
        v6_2.append(score)
    batch_ids = [str(row.get("batch_id") or "") for row in development]
    if "" in batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("#252 sealed batch identity missing or duplicated")
    return development, v6_2


def _load_target_free_batch_rows(
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
    *,
    selected_market_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        actions.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev["five_action_grid"], "#252 five-action grid"
                    )["path"]
                )
            )
        )
        features.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev["feature_rows"], "#252 target-free features"
                    )["path"]
                )
            )
        )
        scored.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        score["mean_ev_scored_rows"], "#252 v6.2 scored rows"
                    )["path"]
                )
            )
        )
    def keep(row: dict[str, Any]) -> bool:
        return str(row.get("market_id") or "") in selected_market_ids

    actions = [row for row in actions if keep(row)]
    features = [row for row in features if keep(row)]
    scored = [row for row in scored if keep(row)]
    if (
        {str(row.get("market_id") or "") for row in actions}
        != selected_market_ids
        or {str(row.get("market_id") or "") for row in features}
        != selected_market_ids
        or {str(row.get("market_id") or "") for row in scored}
        != selected_market_ids
    ):
        raise ValueError("#252 selected market batch coverage is incomplete")
    return actions, features, scored


def _complete_five_action_grid(
    action_rows: list[dict[str, Any]],
    selected_market_ids: set[str],
) -> bool:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in action_rows:
        market_id = str(row.get("market_id") or "")
        if market_id in selected_market_ids:
            groups[(market_id, int(row.get("decision_ts") or 0))].add(
                str(row.get("action") or "")
            )
    markets = {market_id for market_id, _ in groups}
    return (
        markets == selected_market_ids
        and bool(groups)
        and all(actions == FIVE_ACTIONS for actions in groups.values())
    )


def _write_freeze_outputs(
    *,
    run_dir: Path,
    report: dict[str, Any],
    readiness_path: Path,
    gate_plan_path: Path,
    protocol_path: Path,
    index_snapshot: Path,
    v6_7_profile_path: Path,
    selected: list[dict[str, Any]],
    attempted: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    features: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
) -> dict[str, Any]:
    row_outputs = {
        "selected_rows": (
            run_dir / "powered_v6_7_selected_index_rows.jsonl",
            selected,
        ),
        "attempted_rows": (
            run_dir / "powered_v6_7_attempted_index_rows.jsonl",
            attempted,
        ),
        "target_free_five_action_rows": (
            run_dir / "powered_v6_7_five_action_rows.jsonl",
            actions,
        ),
        "target_free_feature_rows": (
            run_dir / "powered_v6_7_feature_rows.jsonl",
            features,
        ),
        "v6_2_scored_rows": (
            run_dir / "powered_v6_7_v6_2_scored_rows.jsonl",
            scored,
        ),
        "v6_7_candidate_rows": (
            run_dir / "powered_v6_7_candidate_rows.jsonl",
            candidates,
        ),
        "v6_7_selected_rows": (
            run_dir / "powered_v6_7_selected_rows.jsonl",
            selected_rows,
        ),
        "v6_7_guard_replay": (
            run_dir / "powered_v6_7_guard_replay.jsonl",
            guard_rows,
        ),
        "retained_v6_7_runtime_decisions": (
            run_dir / "powered_v6_7_runtime_decisions.jsonl",
            runtime_rows,
        ),
    }
    for path, rows in row_outputs.values():
        _write_jsonl(path, rows)
    report_path = run_dir / "powered_v6_7_target_free_freeze_report.json"
    report_md_path = run_dir / "powered_v6_7_target_free_freeze_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# Retained v6.7 Powered Target-Free Freeze",
                "",
                f"- selected markets: `{report['selected_market_count']}`",
                f"- attempted markets: `{report['attempted_market_count']}`",
                f"- guard accepted: `{report['guard_accepted_market_count']}`",
                f"- target-free freeze passed: `{str(report['target_free_freeze_passed']).lower()}`",
                f"- blockers: `{report['target_free_blocking_reason_codes']}`",
                "- outcomes/resolution/labels/PnL opened: `false`",
                "- paper/live/write/wallet/capital remain blocked.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": TARGET_FREE_FREEZE_SCHEMA_VERSION,
        "run_id": report["run_id"],
        "implementation_commit": report["implementation_commit"],
        "decision_freeze_created_ts": report["decision_freeze_created_ts"],
        "exact_market_count": report["selected_market_count"],
        "readiness_manifest": _descriptor(readiness_path),
        "forward_gate_plan": _descriptor(gate_plan_path),
        "collector_protocol": _descriptor(protocol_path),
        "collector_index_snapshot": _descriptor(index_snapshot),
        "v6_7_profile": _descriptor(v6_7_profile_path),
        "development_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in development
        ],
        "v6_2_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in v6_2
        ],
        **{
            name: _descriptor(path)
            for name, (path, _) in row_outputs.items()
        },
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "target_free_freeze_passed": report["target_free_freeze_passed"],
        "target_free_blocking_reason_codes": report[
            "target_free_blocking_reason_codes"
        ],
        "future_target_access_allowed": report["future_target_access_allowed"],
        "decision_freeze_written_before_target_access": True,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_model_cost_sizing_guard_or_gate_tuning_performed": False,
        "paper_candidate_auto_unlock_allowed": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "powered_v6_7_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validated_target_free_freeze(
    path: Path,
    *,
    expected_sha256: str,
    readiness_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _verify_pin(path, expected_sha256, "#252 target-free freeze")
    freeze = _load_json(path)
    if (
        freeze.get("schema_version") != TARGET_FREE_FREEZE_SCHEMA_VERSION
        or freeze.get("exact_market_count") != EXACT_MARKET_COUNT
        or freeze.get("readiness_manifest") != _descriptor(readiness_path)
        or freeze.get("target_free_freeze_passed") is not True
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("decision_freeze_written_before_target_access") is not True
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_scores_mutated") is not False
        or freeze.get(
            "threshold_model_cost_sizing_guard_or_gate_tuning_performed"
        )
        is not False
        or not all(
            freeze.get(field) == expected for field, expected in SAFETY.items()
        )
    ):
        raise ValueError("#252 target-free freeze is not target-access eligible")
    selected = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["selected_rows"], "#252 selected rows"
            )["path"]
        )
    )
    decisions = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["retained_v6_7_runtime_decisions"],
                "#252 retained v6.7 runtime decisions",
            )["path"]
        )
    )
    selected_ids = {str(row.get("market_id") or "") for row in selected}
    decision_ids = {str(row.get("market_id") or "") for row in decisions}
    if (
        len(selected) != EXACT_MARKET_COUNT
        or "" in selected_ids
        or len(selected_ids) != EXACT_MARKET_COUNT
        or len(decisions) < MINIMUM_GUARD_ACCEPTED_MARKET_COUNT
        or "" in decision_ids
        or len(decision_ids) != len(decisions)
        or not decision_ids <= selected_ids
    ):
        raise ValueError("#252 frozen market or decision support is invalid")
    return freeze, selected, decisions


def _settle(
    config: RetainedV67PoweredPaperGateConfig,
    *,
    readiness_path: Path,
    freeze_path: Path,
    freeze: dict[str, Any],
    selected: list[dict[str, Any]],
    provider_factory: Callable[[], Any] | None,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    clock_ms_fn: Callable[[], int],
) -> dict[str, Any]:
    from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider

    if config.stage_started_ts <= max(
        int(freeze["decision_freeze_created_ts"]),
        max(int(row["market_end_ts"]) for row in selected),
    ):
        raise ValueError("#252 target access attempted before freeze or market close")
    claim_path = freeze_path.parent / "powered_v6_7_settlement_single_use_claim.json"
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-single-use-claim-v1",
        "run_id": config.run_id,
        "target_access_started_ts": config.stage_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    _write_single_use_claim(claim_path, claim)
    feature_rows = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["target_free_feature_rows"],
                "#252 frozen target-free features",
            )["path"]
        )
    )
    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        market_id = str(row.get("market_id") or "")
        if (
            not market_id
            or int(row.get("max_input_ts") or 0)
            > int(row.get("decision_ts") or 0)
        ):
            raise ValueError("#252 frozen feature identity or causality invalid")
        frozen_features[market_id].append(row)
    if set(frozen_features) != {str(row["market_id"]) for row in selected}:
        raise ValueError("#252 frozen feature market coverage incomplete")
    run_dir = _prepare_run_dir(
        config.output_dir.resolve(),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker_path = run_dir / "powered_v6_7_settlement_started.json"
    _write_json(marker_path, claim)
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
    attempt_count = 0
    deadline = monotonic_fn() + config.settlement_max_wait_seconds
    while pending:
        attempt_count += 1
        results = _finalize_selected_rounds(
            pending,
            run_dir=run_dir,
            provider_factory=factory,
            max_workers=config.max_workers,
            settlement_attempt=attempt_count,
            evaluation_only_frozen_features_by_market=frozen_features,
        )
        retryable: set[str] = set()
        for result in results:
            market_id = str(result["market_id"])
            if result["settled_corpus_ready"] is True:
                successes[market_id] = result["index_entry"]
                failures.pop(market_id, None)
            else:
                failure = result["failure"]
                failures[market_id] = failure
                if _is_retryable_settlement_failure(failure):
                    retryable.add(market_id)
        if not retryable:
            break
        retried.update(retryable)
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            for market_id in retryable:
                failures[market_id]["reason_codes"] = sorted(
                    {
                        *failures[market_id].get("reason_codes", []),
                        "settlement_max_wait_elapsed",
                    }
                )
            break
        sleep_fn(min(config.settlement_poll_interval_seconds, remaining))
        pending = [
            selected_by_market[market_id] for market_id in sorted(retryable)
        ]
    entries = sorted(successes.values(), key=lambda row: str(row["market_id"]))
    unresolved = sorted(
        (
            failure
            for market_id, failure in failures.items()
            if market_id not in successes
        ),
        key=lambda row: str(row["market_id"]),
    )
    complete = len(entries) == EXACT_MARKET_COUNT and not unresolved
    finalized_ts = int(clock_ms_fn())
    if finalized_ts < config.stage_started_ts:
        raise ValueError("#252 settlement finalized before target access")
    index_path = run_dir / "powered_v6_7_settled_corpus_index.json"
    index: dict[str, Any] | None = None
    if complete:
        index = {
            "schema_version": SETTLED_INDEX_SCHEMA_VERSION,
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "readiness_manifest": _descriptor(readiness_path),
            "target_free_freeze_manifest": _descriptor(freeze_path),
            "target_access_claim": _descriptor(claim_path),
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "entry_count": len(entries),
            "entries": entries,
            "official_read_only_resolution": True,
            "source_outcome_blind_rounds_mutated": False,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "future_results_used_for_tuning": False,
            "result_selected_rerun_allowed": False,
            **SAFETY,
        }
        index["settled_corpus_index_id"] = canonical_json_sha256(index)
        _write_json(index_path, index)
    reasons = Counter(
        str(reason)
        for failure in unresolved
        for reason in failure.get("reason_codes", [])
    )
    blockers = [] if complete else ["powered_settlement_incomplete"]
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-report-v1",
        "run_id": config.run_id,
        "selected_market_count": len(selected),
        "settled_market_count": len(entries),
        "unresolved_or_failed_market_count": len(unresolved),
        "settlement_attempt_count": attempt_count,
        "settlement_retry_market_count": len(retried),
        "unresolved_or_failed_reason_distribution": dict(sorted(reasons.items())),
        "settled_corpus_index_ready": complete,
        "blocking_reason_codes": blockers,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "outcomes_used_for_decision_selection_or_tuning": False,
        "paper_candidate_auto_unlock_allowed": False,
        **SAFETY,
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "powered_v6_7_settlement_report.json"
    report_md_path = run_dir / "powered_v6_7_settlement_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# Retained v6.7 Powered Read-Only Settlement",
                "",
                f"- selected markets: `{len(selected)}`",
                f"- settled markets: `{len(entries)}`",
                f"- unresolved markets: `{len(unresolved)}`",
                f"- settled index ready: `{str(complete).lower()}`",
                f"- blockers: `{blockers}`",
                "- source outcome-blind rounds mutated: `false`",
                "- paper/live/write/wallet/capital remain blocked.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "readiness_manifest": _descriptor(readiness_path),
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settlement_single_use_claim": _descriptor(claim_path),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "powered_v6_7_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    result = _result(run_dir, report, report_path, manifest, manifest_path)
    result.update(
        {
            "index": index,
            "index_path": str(index_path) if complete else None,
            "index_sha256": _sha256_file(index_path) if complete else None,
        }
    )
    return result


def _evaluate(
    config: RetainedV67PoweredPaperGateConfig,
    *,
    readiness_path: Path,
    freeze_path: Path,
    freeze: dict[str, Any],
    selected: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    gate_plan: dict[str, Any],
) -> dict[str, Any]:
    settled_path = Path(config.settled_corpus_index_path).resolve()
    runtime_path = Path(config.runtime_policy_profile_path).resolve()
    _verify_pin(
        settled_path,
        str(config.expected_settled_corpus_index_sha256),
        "#252 settled index",
    )
    _verify_pin(
        runtime_path,
        str(config.expected_runtime_policy_profile_sha256),
        "#252 runtime profile",
    )
    if (
        config.expected_runtime_policy_profile_sha256
        != FROZEN_RUNTIME_POLICY_PROFILE_SHA256
    ):
        raise ValueError("#252 runtime profile pin drifted")
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if (
        runtime_policy_source_hashes()
        != runtime_profile["runtime_policy_contract"]["source_function_sha256"]
    ):
        raise ValueError("#252 runtime policy source hashes drifted")
    settled = _load_json(settled_path)
    entries = _validate_settled_index(
        settled,
        readiness_path=readiness_path,
        freeze_path=freeze_path,
        selected_market_ids=[str(row["market_id"]) for row in selected],
        evaluation_started_ts=config.stage_started_ts,
    )
    claim_path = freeze_path.parent / "powered_v6_7_pnl_gate_single_use_claim.json"
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-gate-single-use-claim-v1",
        "run_id": config.run_id,
        "evaluation_started_ts": config.stage_started_ts,
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_corpus_index": _descriptor(settled_path),
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **SAFETY,
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    _write_single_use_claim(claim_path, claim)
    target_rows = _runtime_targets_for_decisions(
        decisions,
        settled_entries=entries,
        runtime_profile=runtime_profile,
        run_id=config.run_id,
        role="future_powered_paper_candidate_readiness",
    )
    report = build_powered_pnl_gate(
        target_rows,
        evaluation_market_ids=[str(row["market_id"]) for row in selected],
        minimum_guard_accepted_market_count=MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        gate_plan=gate_plan,
        target_free_freeze_sha256=str(
            config.expected_target_free_freeze_manifest_sha256
        ),
    )
    report.update(
        {
            "run_id": config.run_id,
            "implementation_commit": config.implementation_commit,
            "evaluation_started_ts": config.stage_started_ts,
            "complete_official_read_only_settlement": True,
            "pnl_gate_single_use_claim": _descriptor(claim_path),
        }
    )
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    run_dir = _prepare_run_dir(
        config.output_dir.resolve(),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    targets_path = run_dir / "powered_v6_7_runtime_targets.jsonl"
    report_path = run_dir / "powered_v6_7_pnl_gate_report.json"
    report_md_path = run_dir / "powered_v6_7_pnl_gate_report.md"
    _write_jsonl(targets_path, target_rows)
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# Retained v6.7 Powered Paper-Candidate Gate",
                "",
                f"- evaluation markets: `{report['evaluation_market_count']}`",
                f"- guard accepted: `{report['guard_accepted_unique_market_count']}`",
                f"- total after-cost PnL: `{report['total_after_cost_pnl']}`",
                "- largest-winner-removed PnL: "
                f"`{report['largest_winner_removed_after_cost_pnl']}`",
                "- one-sided bootstrap LCB: "
                f"`{report['market_bootstrap']['one_sided_lower_confidence_bound']}`",
                "- powered readiness gate passed: "
                f"`{str(report['powered_paper_candidate_readiness_gate_passed']).lower()}`",
                "- manual paper authorization review eligible: "
                f"`{str(report['manual_paper_authorization_review_eligible']).lower()}`",
                f"- blockers: `{report['powered_paper_candidate_readiness_blocking_reason_codes']}`",
                "- automatic paper/live unlock: `false`",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-pnl-gate-manifest-v1",
        "run_id": config.run_id,
        "readiness_manifest": _descriptor(readiness_path),
        "target_free_freeze_manifest": _descriptor(freeze_path),
        "settled_corpus_index": _descriptor(settled_path),
        "runtime_policy_profile": _descriptor(runtime_path),
        "single_use_claim": _descriptor(claim_path),
        "runtime_targets": _descriptor(targets_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "powered_paper_candidate_readiness_gate_passed": report[
            "powered_paper_candidate_readiness_gate_passed"
        ],
        "powered_paper_candidate_readiness_blocking_reason_codes": report[
            "powered_paper_candidate_readiness_blocking_reason_codes"
        ],
        "manual_paper_authorization_review_eligible": report[
            "manual_paper_authorization_review_eligible"
        ],
        "paper_candidate_auto_unlock_allowed": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        "result_selected_extension_allowed": False,
        **SAFETY,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "powered_v6_7_pnl_gate_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _validate_settled_index(
    index: dict[str, Any],
    *,
    readiness_path: Path,
    freeze_path: Path,
    selected_market_ids: list[str],
    evaluation_started_ts: int,
) -> list[dict[str, Any]]:
    entries = list(index.get("entries") or [])
    entry_ids = [str(row.get("market_id") or "") for row in entries]
    if (
        index.get("schema_version") != SETTLED_INDEX_SCHEMA_VERSION
        or index.get("readiness_manifest") != _descriptor(readiness_path)
        or index.get("target_free_freeze_manifest") != _descriptor(freeze_path)
        or index.get("entry_count") != EXACT_MARKET_COUNT
        or len(entries) != EXACT_MARKET_COUNT
        or set(entry_ids) != set(selected_market_ids)
        or "" in entry_ids
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("official_read_only_resolution") is not True
        or index.get("source_outcome_blind_rounds_mutated") is not False
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
        or index.get("future_results_used_for_tuning") is not False
        or not all(
            index.get(field) == expected for field, expected in SAFETY.items()
        )
    ):
        raise ValueError("#252 settled corpus index is not evaluation eligible")
    _verified_descriptor(index["target_access_claim"], "#252 target access claim")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#252 settled entry violates quarantine contract")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"#252 settled {name}")
    return entries


__all__ = [
    "EXACT_MARKET_COUNT",
    "MAXIMUM_CAPTURE_ATTEMPT_COUNT",
    "MINIMUM_GUARD_ACCEPTED_MARKET_COUNT",
    "RetainedV67PoweredPaperGateConfig",
    "build_powered_pnl_gate",
    "run_retained_v6_7_powered_paper_gate",
    "select_powered_target_free_window",
    "validate_powered_gate_plan",
]
