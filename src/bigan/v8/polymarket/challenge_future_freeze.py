"""Exact-model, target-free challenge future decision freeze.

This module bridges the persistent outcome-blind collector to the parallel
candidate gate.  It deliberately stops at the decision freeze: settlement and
PnL remain inaccessible until all three candidate streams have been written
and hash-pinned.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.canonical_payload import canonical_payload_sha256
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.exact_model_runtime_binding import (
    ExactModelRuntimeBindingConfig,
    verify_exact_model_runtime_binding,
)
from bigan.v8.polymarket.parallel_future_gate import (
    PARALLEL_FREEZE_SCHEMA_VERSION,
    REQUIRED_CANDIDATES,
    build_parallel_target_free_freeze,
    validate_parallel_candidate_protocol,
    validate_parallel_frozen_model_binding,
    validate_parallel_future_collection_plan,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_canary import (
    _score_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout import (
    materialize_adaptive_support_controller_v8_1_runtime_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    _baseline_guard_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_non_risk_abstention_fallback_v8_3 import (
    build_non_risk_abstention_fallback_v8_3_canary,
    validate_non_risk_abstention_fallback_v8_3_profile,
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
from bigan.v8.polymarket.training.execution_layer_v2_v6_7_relative_safe_policy_v7_2 import (
    FORBIDDEN_INFERENCE_FIELDS,
)

CHALLENGE_FUTURE_FREEZE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-target-free-freeze-report-v1"
)
CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-future-target-free-freeze-manifest-v1"
)
SHARED_SOURCE_ROW_SCHEMA_VERSION = "bigan-v8-parallel-shared-source-row-v1"


class ChallengeFutureFreezeError(ValueError):
    """Raised when the challenge decision freeze cannot remain authoritative."""


@dataclass(frozen=True, slots=True)
class ChallengeFutureFreezeConfig:
    """Hash-pinned inputs for the exact-120 parallel candidate freeze."""

    run_id: str
    output_dir: Path | str
    service_root: Path | str
    collection_plan_path: Path | str
    expected_collection_plan_sha256: str
    parallel_protocol_path: Path | str
    expected_parallel_protocol_sha256: str
    v8_1_contract_path: Path | str
    expected_v8_1_contract_sha256: str
    v8_3_contract_path: Path | str
    expected_v8_3_contract_sha256: str
    v6_7_contract_path: Path | str
    expected_v6_7_contract_sha256: str
    frozen_model_binding_path: Path | str
    expected_frozen_model_binding_sha256: str
    historical_gate_contract_path: Path | str
    expected_historical_gate_contract_sha256: str
    historical_replay_report_path: Path | str
    expected_historical_replay_report_sha256: str
    prefreeze_checklist_path: Path | str
    expected_prefreeze_checklist_sha256: str
    excluded_capture_ledger_path: Path | str
    expected_excluded_capture_ledger_sha256: str
    supersession_governance_path: Path | str
    expected_supersession_governance_sha256: str
    historical_fit_manifest_path: Path | str
    expected_historical_fit_manifest_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    v8_3_profile_path: Path | str
    expected_v8_3_profile_sha256: str
    implementation_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if (
            len(self.implementation_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_commit.lower()
            )
        ):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("decision_freeze_created_ts must be positive")
        path_fields = (
            "output_dir",
            "service_root",
            "collection_plan_path",
            "parallel_protocol_path",
            "v8_1_contract_path",
            "v8_3_contract_path",
            "v6_7_contract_path",
            "frozen_model_binding_path",
            "historical_gate_contract_path",
            "historical_replay_report_path",
            "prefreeze_checklist_path",
            "excluded_capture_ledger_path",
            "supersession_governance_path",
            "historical_fit_manifest_path",
            "collector_protocol_path",
            "feature_contract_path",
            "v8_3_profile_path",
        )
        for name in path_fields:
            object.__setattr__(self, name, Path(getattr(self, name)))
        hash_fields = (
            "expected_collection_plan_sha256",
            "expected_parallel_protocol_sha256",
            "expected_v8_1_contract_sha256",
            "expected_v8_3_contract_sha256",
            "expected_v6_7_contract_sha256",
            "expected_frozen_model_binding_sha256",
            "expected_historical_gate_contract_sha256",
            "expected_historical_replay_report_sha256",
            "expected_prefreeze_checklist_sha256",
            "expected_excluded_capture_ledger_sha256",
            "expected_supersession_governance_sha256",
            "expected_historical_fit_manifest_sha256",
            "expected_collector_protocol_sha256",
            "expected_feature_contract_sha256",
            "expected_v8_3_profile_sha256",
        )
        for name in hash_fields:
            _require_sha256(str(getattr(self, name)), name=name)


def select_challenge_future_window(
    index_rows: list[dict[str, Any]],
    *,
    collection_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select the chronological earliest exact target under the frozen cap."""

    collection = dict(collection_plan.get("collection") or {})
    boundary = int(
        collection.get("strictly_later_minimum_market_start_ts_exclusive")
        or 0
    )
    target = int(collection.get("quality_valid_market_target") or 0)
    maximum = int(collection.get("maximum_attempted_market_count") or 0)
    if boundary <= 0 or target != 120 or maximum != 180:
        raise ChallengeFutureFreezeError(
            "challenge collection plan boundary/target/cap is invalid"
        )
    ordered = sorted(
        index_rows,
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("sequence") or 0),
            str(row.get("run_id") or ""),
        ),
    )
    attempted = ordered[:maximum]
    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    selected_market_ids: set[str] = set()
    selected_slugs: set[str] = set()
    selected_decision_ids: set[str] = set()
    selected_source_hashes: set[str] = set()
    for row in attempted:
        reasons: list[str] = []
        market_id = str(row.get("market_id") or "")
        slug = str(row.get("slug") or "")
        decision_id = str(row.get("decision_id") or "")
        source_hash = str(row.get("source_row_hash") or "")
        if row.get("capture_quality_valid") is not True:
            reasons.append("capture_quality_invalid")
        if int(row.get("scheduled_round_start_ts") or 0) <= boundary:
            reasons.append("scheduled_round_not_strictly_later")
        if int(row.get("market_start_ts") or 0) <= boundary:
            reasons.append("market_start_not_strictly_later")
        if not market_id:
            reasons.append("market_id_missing")
        if not slug:
            reasons.append("slug_missing")
        for name, value in (
            ("decision_id", decision_id),
            ("source_row_hash", source_hash),
        ):
            try:
                _require_sha256(value, name=name)
            except ValueError:
                reasons.append(f"{name}_invalid")
        if market_id in selected_market_ids:
            reasons.append("selected_market_id_duplicate")
        if slug in selected_slugs:
            reasons.append("selected_slug_duplicate")
        if decision_id in selected_decision_ids:
            reasons.append("selected_decision_id_duplicate")
        if source_hash in selected_source_hashes:
            reasons.append("selected_source_row_hash_duplicate")
        if reasons:
            exclusions.update(set(reasons))
            continue
        selected.append(row)
        selected_market_ids.add(market_id)
        selected_slugs.add(slug)
        selected_decision_ids.add(decision_id)
        selected_source_hashes.add(source_hash)
        if len(selected) == target:
            break
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    summary = {
        "attempted_scan_count": len(attempted),
        "available_index_entry_count": len(index_rows),
        "selected_market_count": len(selected),
        "target_market_count": target,
        "maximum_attempted_market_count": maximum,
        "remaining_quality_valid_market_count": max(0, target - len(selected)),
        "selected_sequence_start": (
            int(selected[0]["sequence"]) if selected else None
        ),
        "selected_sequence_end": (
            int(selected[-1]["sequence"]) if selected else None
        ),
        "selected_market_ids_sha256": canonical_json_sha256(selected_ids),
        "exclusion_reason_distribution": dict(sorted(exclusions.items())),
        "strictly_later_time_violation_count": sum(
            int(row.get("scheduled_round_start_ts") or 0) <= boundary
            or int(row.get("market_start_ts") or 0) <= boundary
            for row in selected
        ),
        "selected_identity_duplicate_count": (
            len(selected) - len(selected_market_ids)
            + len(selected) - len(selected_slugs)
            + len(selected) - len(selected_decision_ids)
            + len(selected) - len(selected_source_hashes)
        ),
        "attempt_cap_exhausted": (
            len(attempted) >= maximum and len(selected) < target
        ),
        "exact_window_ready": len(selected) == target,
    }
    return selected, attempted, summary


def build_parallel_shared_source_rows(
    selected_index_rows: list[dict[str, Any]],
    *,
    baseline_guard_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one target-free decision-key grid shared by all candidates."""

    baseline_by_market = _one_row_per_market(
        baseline_guard_rows,
        label="matched v6.7 guard",
    )
    output: list[dict[str, Any]] = []
    for index_row in selected_index_rows:
        market_id = str(index_row.get("market_id") or "")
        baseline = baseline_by_market.get(market_id)
        if baseline is None:
            raise ChallengeFutureFreezeError(
                f"matched v6.7 guard missing for {market_id}"
            )
        policy_decision_ts = int(baseline.get("decision_ts") or 0)
        shared_decision_ts = (
            policy_decision_ts
            if policy_decision_ts > 0
            else int(index_row.get("scheduled_round_start_ts") or 0)
        )
        raw_hashes = {
            str(name): str(descriptor.get("sha256") or "")
            for name, descriptor in sorted(
                dict(index_row.get("raw_artifacts") or {}).items()
            )
        }
        if not market_id or shared_decision_ts <= 0:
            raise ChallengeFutureFreezeError(
                "shared source identity or decision timestamp is invalid"
            )
        row = {
            "schema_version": SHARED_SOURCE_ROW_SCHEMA_VERSION,
            "market_id": market_id,
            "slug": str(index_row.get("slug") or ""),
            "decision_ts": shared_decision_ts,
            "scheduled_round_start_ts": int(
                index_row.get("scheduled_round_start_ts") or 0
            ),
            "market_start_ts": int(index_row.get("market_start_ts") or 0),
            "market_end_ts": int(index_row.get("market_end_ts") or 0),
            "collector_sequence": int(index_row.get("sequence") or 0),
            "collector_batch_id": str(index_row.get("batch_id") or ""),
            "collector_entry_sha256": str(
                index_row.get("entry_sha256") or ""
            ),
            "collector_source_row_hash": str(
                index_row.get("source_row_hash") or ""
            ),
            "raw_artifact_sha256s": raw_hashes,
            "capture_quality_valid": True,
            "target_used_as_decision_input": False,
        }
        if policy_decision_ts > 0:
            row["policy_grid_decision_ts"] = policy_decision_ts
        row["shared_source_row_id"] = canonical_payload_sha256(
            row,
            payload_schema_version=SHARED_SOURCE_ROW_SCHEMA_VERSION,
        )
        output.append(row)
    keys = [
        (str(row["market_id"]), int(row["decision_ts"])) for row in output
    ]
    if len(keys) != len(set(keys)):
        raise ChallengeFutureFreezeError(
            "parallel shared source-row grid contains duplicate decision keys"
        )
    return output


def build_challenge_parallel_decisions(
    shared_source_rows: list[dict[str, Any]],
    *,
    v8_1_guard_rows: list[dict[str, Any]],
    v8_3_overlay_rows: list[dict[str, Any]],
    v6_7_guard_rows: list[dict[str, Any]],
    position_size: float,
) -> dict[str, list[dict[str, Any]]]:
    """Project native policy rows onto the exact parallel-gate contract."""

    if position_size <= 0.0:
        raise ChallengeFutureFreezeError("position_size must be positive")
    v81_by_market = _one_row_per_market(
        v8_1_guard_rows,
        label="v8.1 guard",
    )
    v83_by_market = _one_row_per_market(
        v8_3_overlay_rows,
        label="v8.3 overlay",
    )
    v67_by_market = _one_row_per_market(
        v6_7_guard_rows,
        label="matched v6.7 guard",
    )
    output = {candidate_id: [] for candidate_id in REQUIRED_CANDIDATES}
    for source in shared_source_rows:
        market_id = str(source["market_id"])
        decision_ts = int(source["decision_ts"])
        v8_1 = _required_market_row(v81_by_market, market_id, "v8.1")
        v8_3 = _required_market_row(v83_by_market, market_id, "v8.3")
        v6_7 = _required_market_row(v67_by_market, market_id, "v6.7")

        v8_1_action = str(v8_1.get("selected_action") or "NO_TRADE")
        v8_1_allowed = v8_1.get("execution_guard_order_allowed") is True
        v8_1_abstained = v8_1_action == "NO_TRADE"
        output["v8_1_primary_no_fallback"].append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "policy_decision_ts": int(v8_1.get("decision_ts") or 0),
                "executed_action": v8_1_action,
                "selected_side": str(
                    v8_1.get("selected_side")
                    or _side_for_action(v8_1_action)
                ),
                "decision_origin": (
                    "v8_1_primary_abstention"
                    if v8_1_abstained
                    else "v8_1_primary"
                ),
                "fallback_used": False,
                "primary_abstained": v8_1_abstained,
                "execution_guard_order_allowed": v8_1_allowed,
                "execution_blocking_reason_codes": sorted(
                    str(reason)
                    for reason in (
                        v8_1.get("execution_blocking_reason_codes") or []
                    )
                ),
                "proposed_order_size": position_size,
                "full_execution_guard_unchanged": (
                    v8_1.get("full_execution_guard_unchanged") is True
                ),
                "target_used_as_decision_input": False,
            }
        )

        v8_3_action = str(v8_3.get("selected_action") or "NO_TRADE")
        v8_3_source = str(
            v8_3.get("selection_source") or "fail_closed_no_trade"
        )
        output["v8_3_primary_with_fallback"].append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "policy_decision_ts": int(v8_3.get("decision_ts") or 0),
                "executed_action": v8_3_action,
                "selected_side": str(
                    v8_3.get("selected_side")
                    or _side_for_action(v8_3_action)
                ),
                "decision_origin": v8_3_source,
                "fallback_used": v8_3.get("fallback_applied") is True,
                "primary_abstained": (
                    str(v8_3.get("original_v8_1_action") or "")
                    == "NO_TRADE"
                ),
                "execution_guard_order_allowed": (
                    v8_3.get("execution_guard_order_allowed") is True
                ),
                "selection_reason_codes": sorted(
                    str(reason)
                    for reason in (
                        v8_3.get("selection_reason_codes") or []
                    )
                ),
                "proposed_order_size": position_size,
                "full_execution_guard_unchanged": (
                    v8_3.get("full_execution_guard_unchanged") is True
                ),
                "v8_3_frozen_contract_reproduced": True,
                "target_used_as_decision_input": False,
            }
        )

        v6_7_action = str(v6_7.get("selected_action") or "NO_TRADE")
        output["matched_frozen_v6_7"].append(
            {
                "market_id": market_id,
                "decision_ts": decision_ts,
                "policy_decision_ts": int(v6_7.get("decision_ts") or 0),
                "executed_action": v6_7_action,
                "selected_side": str(
                    v6_7.get("selected_side")
                    or _side_for_action(v6_7_action)
                ),
                "decision_origin": (
                    "matched_v6_7_abstention"
                    if v6_7_action == "NO_TRADE"
                    else "matched_v6_7_primary"
                ),
                "fallback_used": False,
                "primary_abstained": v6_7_action == "NO_TRADE",
                "execution_guard_order_allowed": (
                    v6_7.get("execution_guard_order_allowed") is True
                ),
                "execution_blocking_reason_codes": sorted(
                    str(reason)
                    for reason in (
                        v6_7.get("execution_blocking_reason_codes") or []
                    )
                ),
                "proposed_order_size": position_size,
                "full_execution_guard_unchanged": (
                    v6_7.get("full_execution_guard_unchanged") is True
                ),
                "matched_baseline_frozen_contract_reproduced": True,
                "target_used_as_decision_input": False,
            }
        )
    return output


def challenge_collection_status(
    *,
    service_root: Path | str,
    collection_plan: dict[str, Any],
) -> dict[str, Any]:
    """Return a read-only readiness snapshot for the running collector."""

    root = Path(service_root).resolve()
    index_path = root / "persistent_outcome_blind_round_index.jsonl"
    rows = load_and_validate_persistent_outcome_blind_index(index_path)
    _, _, selection = select_challenge_future_window(
        rows,
        collection_plan=collection_plan,
    )
    state_path = root / "persistent_outcome_blind_service_state.json"
    state = _load_json(state_path) if state_path.is_file() else {}
    return {
        "service_root": str(root),
        "collector_index_path": str(index_path),
        "collector_index_exists": index_path.is_file(),
        "collector_index_sha256": (
            _sha256_file(index_path) if index_path.is_file() else None
        ),
        "service_status": state.get("status", "batch_in_progress"),
        "last_completed_batch_sequence": int(
            state.get("last_completed_batch_sequence") or 0
        ),
        "quality_valid_index_entry_count": sum(
            row.get("capture_quality_valid") is True for row in rows
        ),
        **selection,
        "labels_outcomes_or_pnl_opened": False,
    }


def run_challenge_future_target_free_freeze(
    config: ChallengeFutureFreezeConfig,
) -> dict[str, Any]:
    """Score and freeze all candidates without opening any future target."""

    paths = {
        "plan": config.collection_plan_path.resolve(),
        "protocol": config.parallel_protocol_path.resolve(),
        "v8_1_contract": config.v8_1_contract_path.resolve(),
        "v8_3_contract": config.v8_3_contract_path.resolve(),
        "v6_7_contract": config.v6_7_contract_path.resolve(),
        "binding": config.frozen_model_binding_path.resolve(),
        "historical_gate_contract": config.historical_gate_contract_path.resolve(),
        "historical_report": config.historical_replay_report_path.resolve(),
        "prefreeze_checklist": config.prefreeze_checklist_path.resolve(),
        "excluded_capture_ledger": (
            config.excluded_capture_ledger_path.resolve()
        ),
        "supersession_governance": (
            config.supersession_governance_path.resolve()
        ),
        "historical_fit_manifest": config.historical_fit_manifest_path.resolve(),
        "collector_protocol": config.collector_protocol_path.resolve(),
        "feature_contract": config.feature_contract_path.resolve(),
        "v8_3_profile": config.v8_3_profile_path.resolve(),
    }
    pins = {
        "plan": config.expected_collection_plan_sha256,
        "protocol": config.expected_parallel_protocol_sha256,
        "v8_1_contract": config.expected_v8_1_contract_sha256,
        "v8_3_contract": config.expected_v8_3_contract_sha256,
        "v6_7_contract": config.expected_v6_7_contract_sha256,
        "binding": config.expected_frozen_model_binding_sha256,
        "historical_gate_contract": (
            config.expected_historical_gate_contract_sha256
        ),
        "historical_report": (
            config.expected_historical_replay_report_sha256
        ),
        "prefreeze_checklist": (
            config.expected_prefreeze_checklist_sha256
        ),
        "excluded_capture_ledger": (
            config.expected_excluded_capture_ledger_sha256
        ),
        "supersession_governance": (
            config.expected_supersession_governance_sha256
        ),
        "historical_fit_manifest": (
            config.expected_historical_fit_manifest_sha256
        ),
        "collector_protocol": config.expected_collector_protocol_sha256,
        "feature_contract": config.expected_feature_contract_sha256,
        "v8_3_profile": config.expected_v8_3_profile_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"challenge future {name}")

    plan = _load_json(paths["plan"])
    protocol = _load_json(paths["protocol"])
    candidate_contracts = {
        "v8_1_primary_no_fallback": _load_json(paths["v8_1_contract"]),
        "v8_3_primary_with_fallback": _load_json(paths["v8_3_contract"]),
        "matched_frozen_v6_7": _load_json(paths["v6_7_contract"]),
    }
    candidate_hashes = {
        "v8_1_primary_no_fallback": pins["v8_1_contract"].lower(),
        "v8_3_primary_with_fallback": pins["v8_3_contract"].lower(),
        "matched_frozen_v6_7": pins["v6_7_contract"].lower(),
    }
    binding = _load_json(paths["binding"])
    historical_report = _load_json(paths["historical_report"])
    prefreeze_checklist = _load_json(paths["prefreeze_checklist"])
    excluded_capture_ledger = _load_json(paths["excluded_capture_ledger"])
    supersession_governance = _load_json(
        paths["supersession_governance"]
    )
    fit_manifest = _load_json(paths["historical_fit_manifest"])
    v8_3_profile = _load_json(paths["v8_3_profile"])
    config_dir = paths["plan"].parent
    validate_parallel_candidate_protocol(
        protocol,
        candidate_contracts=candidate_contracts,
    )
    validate_parallel_frozen_model_binding(
        binding,
        candidate_contracts=candidate_contracts,
        expected_binding_sha256=pins["binding"].lower(),
    )
    validate_parallel_future_collection_plan(
        plan,
        plan_sha256=pins["plan"].lower(),
        protocol_sha256=pins["protocol"].lower(),
        candidate_contract_sha256s=candidate_hashes,
        collector_protocol_sha256=pins["collector_protocol"].lower(),
        feature_contract_sha256=pins["feature_contract"].lower(),
        feature_missingness_contract_sha256=_sha256_file(
            config_dir / "feature_missingness_contract.json"
        ),
        feature_missingness_runtime_schema_sha256=_sha256_file(
            config_dir / "feature_missingness_runtime.schema.json"
        ),
        promotion_evidence_protocol_sha256=_sha256_file(
            config_dir / "challenge_promotion_evidence_protocol.json"
        ),
        frozen_model_binding_sha256=pins["binding"].lower(),
        frozen_model_binding=binding,
        candidate_contracts=candidate_contracts,
        prefreeze_checklist_sha256=pins["prefreeze_checklist"].lower(),
        prefreeze_checklist=prefreeze_checklist,
        excluded_capture_ledger_sha256=pins[
            "excluded_capture_ledger"
        ].lower(),
        excluded_capture_ledger=excluded_capture_ledger,
        historical_gate_contract_sha256=pins[
            "historical_gate_contract"
        ].lower(),
        historical_replay_report_sha256=pins["historical_report"].lower(),
        historical_replay_report=historical_report,
        supersession_governance=supersession_governance,
        supersession_governance_sha256=pins[
            "supersession_governance"
        ].lower(),
        expected_supersession_governance_sha256=pins[
            "supersession_governance"
        ].lower(),
    )
    if (
        str(plan["collection"]["service_root"])
        != _relative_service_root(config.service_root)
    ):
        raise ChallengeFutureFreezeError(
            "service root does not match the frozen collection plan"
        )

    artifacts = _load_and_validate_exact_model_artifacts(
        fit_manifest=fit_manifest,
        binding=binding,
        binding_path=paths["binding"],
        binding_sha256=pins["binding"].lower(),
        v8_3_profile=v8_3_profile,
        v8_3_profile_sha256=pins["v8_3_profile"].lower(),
        candidate_contracts=candidate_contracts,
        v8_1_contract_path=paths["v8_1_contract"],
        v8_1_contract_sha256=pins["v8_1_contract"].lower(),
    )
    index_path = (
        config.service_root.resolve()
        / "persistent_outcome_blind_round_index.jsonl"
    )
    if not index_path.is_file():
        raise ChallengeFutureFreezeError(
            "collector index is not available; collection is still in its first batch"
        )
    source_index_sha256 = _sha256_file(index_path)
    index_rows = load_and_validate_persistent_outcome_blind_index(index_path)
    selected, attempted, selection = select_challenge_future_window(
        index_rows,
        collection_plan=plan,
    )
    if selection["exact_window_ready"] is not True:
        raise ChallengeFutureFreezeError(
            "exact-120 challenge future window is not ready: "
            f"{selection['selected_market_count']}/120"
        )
    for row in selected:
        _verify_index_raw_descriptors(row)

    development, v6_2 = _discover_and_validate_batch_manifests(
        service_root=config.service_root.resolve(),
        selected_index_rows=selected,
        expected_feature_contract_sha256=pins["feature_contract"].lower(),
        expected_v6_2_candidate_sha256=str(
            candidate_contracts["matched_frozen_v6_7"][
                "frozen_v6_2_candidate_manifest_sha256"
            ]
        ).lower(),
        expected_runtime_binding_summary_id=artifacts[
            "runtime_binding_summary"
        ]["summary_id"],
    )
    action_rows, feature_rows, scored_rows = _load_target_free_batch_rows(
        development,
        v6_2,
    )
    selected_ids = [str(row["market_id"]) for row in selected]
    selected_set = set(selected_ids)
    action_rows = [
        row
        for row in action_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    feature_rows = [
        row
        for row in feature_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    scored_rows = [
        row
        for row in scored_rows
        if str(row.get("market_id") or "") in selected_set
    ]
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(
            _find_nonempty_fields(feature_rows, FORBIDDEN_INFERENCE_FIELDS)
        )
        | set(
            _find_nonempty_fields(scored_rows, FORBIDDEN_INFERENCE_FIELDS)
        )
    )
    if forbidden:
        raise ChallengeFutureFreezeError(
            "target-free challenge inputs contain target fields: "
            + ",".join(forbidden)
        )

    v6_7_candidates, v6_7_candidate_summary = (
        build_v6_7_target_free_candidate_rows(
            scored_rows,
            action_rows=action_rows,
            profile=artifacts["v6_7_profile"],
        )
    )
    baseline_rows = select_v6_7_target_free_rows(
        v6_7_candidates,
        profile=artifacts["v6_7_profile"],
    )
    canonical_rows, canonical_summary = (
        _canonicalize_target_free_sbc_rows(
            scored_rows,
            action_rows=action_rows,
            v6_7_profile=artifacts["v6_7_profile"],
            v7_0_profile=artifacts["v7_0_profile"],
        )
    )
    v8_1_decisions, v8_1_guard, final_state = _score_window(
        selected_ids,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=artifacts["model"],
        v6_7_profile=artifacts["v6_7_profile"],
    )
    v6_7_guard = _baseline_guard_window(
        selected_ids,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        v6_7_profile=artifacts["v6_7_profile"],
    )
    overlay = build_non_risk_abstention_fallback_v8_3_canary(
        candidate_rows=v8_1_guard,
        baseline_rows=v6_7_guard,
        profile=v8_3_profile,
    )
    v8_3_overlay = list(overlay["decisions"])
    v8_1_runtime = (
        materialize_adaptive_support_controller_v8_1_runtime_decisions(
            v8_1_guard,
            action_rows=action_rows,
        )
    )
    v6_7_runtime = (
        materialize_adaptive_support_controller_v8_1_runtime_decisions(
            v6_7_guard,
            action_rows=action_rows,
        )
    )
    shared_source_rows = build_parallel_shared_source_rows(
        selected,
        baseline_guard_rows=v6_7_guard,
    )
    latest_collected_market_end_ts = max(
        int(row["market_end_ts"]) for row in shared_source_rows
    )
    if config.decision_freeze_created_ts <= latest_collected_market_end_ts:
        raise ChallengeFutureFreezeError(
            "decision freeze timestamp must follow the complete raw window"
        )
    decisions_by_candidate = build_challenge_parallel_decisions(
        shared_source_rows,
        v8_1_guard_rows=v8_1_guard,
        v8_3_overlay_rows=v8_3_overlay,
        v6_7_guard_rows=v6_7_guard,
        position_size=float(
            binding["execution"]["paper_position_size"]
        ),
    )
    parallel_freeze = build_parallel_target_free_freeze(
        protocol=protocol,
        candidate_contracts=candidate_contracts,
        source_rows=shared_source_rows,
        decisions_by_candidate=decisions_by_candidate,
        decision_freeze_created_ts=config.decision_freeze_created_ts,
        target_access_started=False,
    )
    if _sha256_file(index_path) != source_index_sha256:
        raise ChallengeFutureFreezeError(
            "collector index changed while the challenge freeze was built"
        )

    run_dir = _prepare_run_dir(
        Path(config.output_dir),
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    index_snapshot = run_dir / "challenge_collector_index_snapshot.jsonl"
    shutil.copyfile(index_path, index_snapshot)
    if _sha256_file(index_snapshot) != source_index_sha256:
        raise ChallengeFutureFreezeError(
            "collector index snapshot hash mismatch"
        )
    report = _build_freeze_report(
        run_id=config.run_id,
        implementation_commit=config.implementation_commit,
        decision_freeze_created_ts=config.decision_freeze_created_ts,
        plan=plan,
        selection=selection,
        source_index_sha256=source_index_sha256,
        shared_source_rows=shared_source_rows,
        decisions_by_candidate=decisions_by_candidate,
        parallel_freeze=parallel_freeze,
        v6_7_candidate_summary=v6_7_candidate_summary,
        canonical_summary=canonical_summary,
        final_state=final_state,
        binding=binding,
        runtime_binding_summary=artifacts["runtime_binding_summary"],
    )
    return _write_freeze_outputs(
        run_dir=run_dir,
        report=report,
        parallel_freeze=parallel_freeze,
        index_snapshot=index_snapshot,
        input_paths=paths,
        selected=selected,
        attempted=attempted,
        action_rows=action_rows,
        feature_rows=feature_rows,
        scored_rows=scored_rows,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        v8_1_decisions=v8_1_decisions,
        v8_1_guard=v8_1_guard,
        v8_3_overlay=v8_3_overlay,
        v6_7_guard=v6_7_guard,
        v8_1_runtime=v8_1_runtime,
        v6_7_runtime=v6_7_runtime,
        shared_source_rows=shared_source_rows,
        decisions_by_candidate=decisions_by_candidate,
        final_state=final_state,
        development=development,
        v6_2=v6_2,
        artifact_descriptors=artifacts["descriptors"],
    )


def _load_and_validate_exact_model_artifacts(
    *,
    fit_manifest: dict[str, Any],
    binding: dict[str, Any],
    binding_path: Path,
    binding_sha256: str,
    v8_3_profile: dict[str, Any],
    v8_3_profile_sha256: str,
    candidate_contracts: dict[str, dict[str, Any]],
    v8_1_contract_path: Path,
    v8_1_contract_sha256: str,
) -> dict[str, Any]:
    model_descriptor = _verified_descriptor(
        fit_manifest.get("model"),
        "challenge exact frozen model",
    )
    v8_1_profile_descriptor = _verified_descriptor(
        fit_manifest.get("profile"),
        "challenge v8.1 profile",
    )
    v6_7_descriptor = _verified_descriptor(
        fit_manifest.get("v6_7_candidate_profile"),
        "challenge v6.7 profile",
    )
    v7_0_descriptor = _verified_descriptor(
        fit_manifest.get("v7_0_training_profile"),
        "challenge v7.0 profile",
    )
    training_descriptor = _verified_descriptor(
        fit_manifest.get("seed_runtime_target_rows"),
        "challenge source training rows",
    )
    runtime_binding_summary = verify_exact_model_runtime_binding(
        ExactModelRuntimeBindingConfig(
            candidate_contract_path=v8_1_contract_path,
            expected_candidate_contract_sha256=v8_1_contract_sha256,
            frozen_model_binding_path=binding_path,
            expected_frozen_model_binding_sha256=binding_sha256,
            frozen_model_artifact_path=Path(model_descriptor["path"]),
            expected_frozen_model_artifact_sha256=str(
                model_descriptor["sha256"]
            ),
            candidate_profile_path=Path(v8_1_profile_descriptor["path"]),
            expected_candidate_profile_sha256=str(
                v8_1_profile_descriptor["sha256"]
            ),
        )
    )
    model = _load_json(Path(model_descriptor["path"]))
    v8_1_profile = _load_json(Path(v8_1_profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    v81.validate_adaptive_support_controller_v8_1_profile(v8_1_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    validate_non_risk_abstention_fallback_v8_3_profile(v8_3_profile)
    state = dict(model.get("final_rank_state") or {})
    expected_state = dict(binding.get("initial_controller_state") or {})
    blockers: list[str] = []
    if model_descriptor["sha256"] != binding.get(
        "frozen_model_artifact_sha256"
    ):
        blockers.append("frozen_model_artifact_sha256")
    if model.get("model_artifact_id") != binding.get(
        "frozen_model_artifact_id"
    ):
        blockers.append("frozen_model_artifact_id")
    if (
        (model.get("final_weighted_model") or {}).get("booster_sha256")
        != binding.get("frozen_booster_sha256")
    ):
        blockers.append("frozen_booster_sha256")
    if v8_1_profile_descriptor["sha256"] != binding.get(
        "frozen_profile_sha256"
    ):
        blockers.append("frozen_profile_sha256")
    if v8_1_profile_descriptor["sha256"] != candidate_contracts[
        "v8_1_primary_no_fallback"
    ].get("profile_sha256"):
        blockers.append("v8_1_contract_profile_sha256")
    if training_descriptor["sha256"] != binding.get(
        "source_training_rows_sha256"
    ):
        blockers.append("source_training_rows_sha256")
    for candidate_id in (
        "v8_1_primary_no_fallback",
        "v8_3_primary_with_fallback",
    ):
        if training_descriptor["sha256"] != candidate_contracts[
            candidate_id
        ].get("source_model_hash"):
            blockers.append(f"{candidate_id}:source_model_hash")
    for field in (
        "rank_state_id",
        "rank_lineage_hash",
        "eligible_prediction_scores_hash",
        "controller_guard_acceptance_history_hash",
    ):
        if state.get(field) != expected_state.get(field):
            blockers.append(f"initial_controller_state:{field}")
    if v8_3_profile_sha256 != candidate_contracts[
        "v8_3_primary_with_fallback"
    ].get("profile_sha256"):
        blockers.append("v8_3_profile_sha256")
    if v6_7_descriptor["sha256"] != candidate_contracts[
        "matched_frozen_v6_7"
    ].get("profile_sha256"):
        blockers.append("v6_7_profile_sha256")
    if v6_7_descriptor["sha256"] != candidate_contracts[
        "v8_3_primary_with_fallback"
    ].get("fallback_profile_sha256"):
        blockers.append("v8_3_fallback_profile_sha256")
    if (
        model.get("frozen") is not True
        or model.get("decision_time_safe") is not True
        or model.get("target_free_canary_collection_allowed") is not True
        or model.get("promotion_evidence_eligible") is not False
        or model.get("paper_candidate_allowed") is not False
        or model.get("live_trading_enabled") is not False
        or model.get("polymarket_write_enabled") is not False
        or model.get("wallet_signing_enabled") is not False
        or model.get("capital_at_risk") is not False
    ):
        blockers.append("frozen_model_safety_contract")
    if blockers:
        raise ChallengeFutureFreezeError(
            "exact challenge model binding is invalid: "
            + ",".join(sorted(blockers))
        )
    return {
        "model": model,
        "v8_1_profile": v8_1_profile,
        "v6_7_profile": v6_7_profile,
        "v7_0_profile": v7_0_profile,
        "runtime_binding_summary": runtime_binding_summary,
        "descriptors": {
            "model": model_descriptor,
            "v8_1_profile": v8_1_profile_descriptor,
            "v6_7_profile": v6_7_descriptor,
            "v7_0_profile": v7_0_descriptor,
            "source_training_rows": training_descriptor,
        },
    }


def _discover_and_validate_batch_manifests(
    *,
    service_root: Path,
    selected_index_rows: list[dict[str, Any]],
    expected_feature_contract_sha256: str,
    expected_v6_2_candidate_sha256: str,
    expected_runtime_binding_summary_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_batch_ids = {
        str(row.get("batch_id") or "") for row in selected_index_rows
    }
    if "" in selected_batch_ids:
        raise ChallengeFutureFreezeError("selected collector batch id missing")
    development_by_batch = _manifest_map(
        service_root.glob(
            "batch_canary_runs/*/"
            "execution_layer_v2_outcome_blind_batch_canary_manifest.json"
        ),
        label="development batch",
    )
    v6_2_by_batch = _manifest_map(
        service_root.glob(
            "v6_2_batch_canary_runs/*/"
            "v6_2_future_batch_action_canary_manifest.json"
        ),
        label="v6.2 batch",
    )
    development: list[dict[str, Any]] = []
    v6_2: list[dict[str, Any]] = []
    for batch_id in sorted(selected_batch_ids):
        dev_pair = development_by_batch.get(batch_id)
        score_pair = v6_2_by_batch.get(batch_id)
        if dev_pair is None or score_pair is None:
            raise ChallengeFutureFreezeError(
                f"target-free batch evidence missing for {batch_id}"
            )
        dev_path, dev = dev_pair
        score_path, score = score_pair
        dev_sha = _sha256_file(dev_path)
        feature = _verified_descriptor(
            dev.get("feature_contract"),
            "challenge batch feature contract",
        )
        matched_dev = _verified_descriptor(
            score.get("development_batch_canary_manifest"),
            "challenge matched development batch",
        )
        source_candidate = _verified_descriptor(
            score.get("candidate_manifest"),
            "challenge frozen v6.2 candidate",
        )
        blockers = []
        if dev.get("development_data_canary_passed") is not True:
            blockers.append("development_data_canary")
        if dev.get("candidate_model_scoring_attempted") is not False:
            blockers.append("development_candidate_scoring")
        if dev.get("labels_outcomes_or_pnl_opened") is not False:
            blockers.append("development_target_access")
        runtime_summary = dict(
            dev.get("exact_model_runtime_binding_summary") or {}
        )
        if (
            dev.get("exact_model_runtime_binding_required") is not True
            or dev.get("exact_model_runtime_binding_verified") is not True
            or runtime_summary.get("summary_id")
            != expected_runtime_binding_summary_id
        ):
            blockers.append("exact_model_runtime_binding")
        if feature["sha256"] != expected_feature_contract_sha256:
            blockers.append("feature_contract_sha256")
        if score.get("labels_outcomes_or_pnl_opened") is not False:
            blockers.append("v6_2_target_access")
        if matched_dev["sha256"] != dev_sha:
            blockers.append("matched_development_manifest_sha256")
        if source_candidate["sha256"] != expected_v6_2_candidate_sha256:
            blockers.append("v6_2_candidate_manifest_sha256")
        if blockers:
            raise ChallengeFutureFreezeError(
                f"batch lineage invalid for {batch_id}: "
                + ",".join(blockers)
            )
        dev["_manifest_path"] = str(dev_path)
        score["_manifest_path"] = str(score_path)
        development.append(dev)
        v6_2.append(score)
    return development, v6_2


def _manifest_map(
    paths: Any,
    *,
    label: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    output: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(Path(item).resolve() for item in paths):
        manifest = _load_json(path)
        batch_id = str(manifest.get("batch_id") or "")
        if not batch_id:
            raise ChallengeFutureFreezeError(f"{label} id missing: {path}")
        if batch_id in output:
            raise ChallengeFutureFreezeError(
                f"duplicate {label} manifest for {batch_id}"
            )
        output[batch_id] = (path, manifest)
    return output


def _load_target_free_batch_rows(
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        actions.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev.get("five_action_grid"),
                        "challenge five-action grid",
                    )["path"]
                )
            )
        )
        features.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev.get("feature_rows"),
                        "challenge feature rows",
                    )["path"]
                )
            )
        )
        scored.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        score.get("mean_ev_scored_rows"),
                        "challenge mean-EV scored rows",
                    )["path"]
                )
            )
        )
    return actions, features, scored


def _build_freeze_report(
    *,
    run_id: str,
    implementation_commit: str,
    decision_freeze_created_ts: int,
    plan: dict[str, Any],
    selection: dict[str, Any],
    source_index_sha256: str,
    shared_source_rows: list[dict[str, Any]],
    decisions_by_candidate: dict[str, list[dict[str, Any]]],
    parallel_freeze: dict[str, Any],
    v6_7_candidate_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
    final_state: dict[str, Any],
    binding: dict[str, Any],
    runtime_binding_summary: dict[str, Any],
) -> dict[str, Any]:
    source_keys = [
        (str(row["market_id"]), int(row["decision_ts"]))
        for row in shared_source_rows
    ]
    decision_counts = {
        candidate_id: len(rows)
        for candidate_id, rows in decisions_by_candidate.items()
    }
    accepted_counts = {
        candidate_id: sum(
            row.get("execution_guard_order_allowed") is True
            and row.get("executed_action") != "NO_TRADE"
            for row in rows
        )
        for candidate_id, rows in decisions_by_candidate.items()
    }
    checks = {
        "exact_120_chronological_quality_valid_markets": (
            selection.get("exact_window_ready") is True
            and len(shared_source_rows) == 120
        ),
        "strictly_later_boundary_passed": (
            selection.get("strictly_later_time_violation_count") == 0
        ),
        "identity_disjointness_passed": (
            selection.get("selected_identity_duplicate_count") == 0
        ),
        "same_source_grid_for_all_candidates": (
            len(source_keys) == len(set(source_keys)) == 120
            and all(count == 120 for count in decision_counts.values())
        ),
        "exact_frozen_model_and_initial_controller_state_bound": (
            final_state.get("rank_state_id") is not None
            and binding.get("frozen_model_artifact_id")
            is not None
            and runtime_binding_summary.get(
                "runtime_byte_verification_passed"
            )
            is True
        ),
        "parallel_freeze_hash_written": (
            parallel_freeze.get("schema_version")
            == PARALLEL_FREEZE_SCHEMA_VERSION
            and bool(parallel_freeze.get("freeze_sha256"))
        ),
        "all_candidate_decisions_frozen_before_target_access": (
            parallel_freeze.get(
                "all_candidate_decisions_frozen_before_target_access"
            )
            is True
        ),
        "outcomes_labels_settlement_returns_or_pnl_sealed": (
            parallel_freeze.get(
                "outcomes_labels_settlement_returns_or_pnl_opened"
            )
            is False
        ),
        "safety_unlocks_remain_false": all(
            parallel_freeze.get(field) is False
            for field in (
                "paper_candidate_unlocked",
                "promotion_unlocked",
                "live_unlocked",
                "write_enabled",
                "wallet_enabled",
                "capital_at_risk",
            )
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    report = {
        "schema_version": CHALLENGE_FUTURE_FREEZE_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "fresh_attempt_id": plan.get("fresh_attempt_id"),
        "implementation_commit": implementation_commit,
        "decision_freeze_created_ts": decision_freeze_created_ts,
        "collector_index_sha256": source_index_sha256,
        "selection": selection,
        "shared_source_row_count": len(shared_source_rows),
        "shared_source_rows_sha256": (
            parallel_freeze["shared_source_rows_sha256"]
        ),
        "parallel_freeze_sha256": parallel_freeze["freeze_sha256"],
        "candidate_decision_counts": decision_counts,
        "candidate_guard_accepted_counts": accepted_counts,
        "v6_7_candidate_summary": v6_7_candidate_summary,
        "canonical_mapping_summary": canonical_summary,
        "final_controller_state_id": final_state.get("rank_state_id"),
        "frozen_model_artifact_sha256": binding.get(
            "frozen_model_artifact_sha256"
        ),
        "frozen_model_artifact_id": binding.get(
            "frozen_model_artifact_id"
        ),
        "exact_model_runtime_binding_summary": runtime_binding_summary,
        "target_free_checks": checks,
        "parallel_target_free_freeze_passed": not blockers,
        "target_free_blocking_reason_codes": blockers,
        "future_target_access_allowed": not blockers,
        "result_selected_extension_or_rerun_allowed": False,
        "outcomes_labels_settlement_returns_or_pnl_opened": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    report["report_id"] = canonical_payload_sha256(
        report,
        payload_schema_version=CHALLENGE_FUTURE_FREEZE_REPORT_SCHEMA_VERSION,
    )
    return report


def _write_freeze_outputs(
    *,
    run_dir: Path,
    report: dict[str, Any],
    parallel_freeze: dict[str, Any],
    index_snapshot: Path,
    input_paths: dict[str, Path],
    selected: list[dict[str, Any]],
    attempted: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    v8_1_decisions: list[dict[str, Any]],
    v8_1_guard: list[dict[str, Any]],
    v8_3_overlay: list[dict[str, Any]],
    v6_7_guard: list[dict[str, Any]],
    v8_1_runtime: list[dict[str, Any]],
    v6_7_runtime: list[dict[str, Any]],
    shared_source_rows: list[dict[str, Any]],
    decisions_by_candidate: dict[str, list[dict[str, Any]]],
    final_state: dict[str, Any],
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
    artifact_descriptors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    outputs = {
        "selected_index_rows": run_dir / "challenge_selected_index_rows.jsonl",
        "attempted_index_rows": run_dir / "challenge_attempted_index_rows.jsonl",
        "action_rows": run_dir / "challenge_five_action_rows.jsonl",
        "feature_rows": run_dir / "challenge_feature_rows.jsonl",
        "scored_rows": run_dir / "challenge_v6_2_scored_rows.jsonl",
        "canonical_rows": run_dir / "challenge_canonical_sbc_rows.jsonl",
        "baseline_rows": run_dir / "challenge_v6_7_baseline_rows.jsonl",
        "v8_1_native_decisions": run_dir / "challenge_v8_1_native_decisions.jsonl",
        "v8_1_guard": run_dir / "challenge_v8_1_guard_rows.jsonl",
        "v8_3_overlay": run_dir / "challenge_v8_3_overlay_rows.jsonl",
        "v6_7_guard": run_dir / "challenge_v6_7_guard_rows.jsonl",
        "v8_1_runtime": run_dir / "challenge_v8_1_runtime_rows.jsonl",
        "v6_7_runtime": run_dir / "challenge_v6_7_runtime_rows.jsonl",
        "shared_source_rows": run_dir / "challenge_shared_source_rows.jsonl",
        "v8_1_parallel_decisions": (
            run_dir / "challenge_v8_1_parallel_decisions.jsonl"
        ),
        "v8_3_parallel_decisions": (
            run_dir / "challenge_v8_3_parallel_decisions.jsonl"
        ),
        "v6_7_parallel_decisions": (
            run_dir / "challenge_v6_7_parallel_decisions.jsonl"
        ),
    }
    rows_by_name = {
        "selected_index_rows": selected,
        "attempted_index_rows": attempted,
        "action_rows": action_rows,
        "feature_rows": feature_rows,
        "scored_rows": scored_rows,
        "canonical_rows": canonical_rows,
        "baseline_rows": baseline_rows,
        "v8_1_native_decisions": v8_1_decisions,
        "v8_1_guard": v8_1_guard,
        "v8_3_overlay": v8_3_overlay,
        "v6_7_guard": v6_7_guard,
        "v8_1_runtime": v8_1_runtime,
        "v6_7_runtime": v6_7_runtime,
        "shared_source_rows": shared_source_rows,
        "v8_1_parallel_decisions": decisions_by_candidate[
            "v8_1_primary_no_fallback"
        ],
        "v8_3_parallel_decisions": decisions_by_candidate[
            "v8_3_primary_with_fallback"
        ],
        "v6_7_parallel_decisions": decisions_by_candidate[
            "matched_frozen_v6_7"
        ],
    }
    for name, rows in rows_by_name.items():
        _write_jsonl(outputs[name], rows)
    state_path = run_dir / "challenge_final_controller_state.json"
    freeze_path = run_dir / "challenge_parallel_target_free_freeze.json"
    report_path = run_dir / "challenge_target_free_freeze_report.json"
    report_md_path = run_dir / "challenge_target_free_freeze_report.md"
    _write_json(state_path, final_state)
    _write_json(freeze_path, parallel_freeze)
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# Challenge Parallel Target-Free Freeze",
                "",
                f"- selected markets: `{report['shared_source_row_count']}`",
                f"- accepted support: `{report['candidate_guard_accepted_counts']}`",
                f"- collector index: `{report['collector_index_sha256']}`",
                f"- parallel freeze: `{report['parallel_freeze_sha256']}`",
                "- target-free freeze passed: "
                f"`{str(report['parallel_target_free_freeze_passed']).lower()}`",
                f"- blockers: `{report['target_free_blocking_reason_codes']}`",
                "- outcomes/labels/settlement/PnL opened: `false`",
                "- paper/live/write/wallet/capital remain blocked.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION,
        "run_id": report["run_id"],
        "fresh_attempt_id": report["fresh_attempt_id"],
        "implementation_commit": report["implementation_commit"],
        "decision_freeze_created_ts": report["decision_freeze_created_ts"],
        "exact_market_count": report["shared_source_row_count"],
        "collection_plan": _descriptor(input_paths["plan"]),
        "parallel_protocol": _descriptor(input_paths["protocol"]),
        "v8_1_candidate_contract": _descriptor(
            input_paths["v8_1_contract"]
        ),
        "v8_3_candidate_contract": _descriptor(
            input_paths["v8_3_contract"]
        ),
        "v6_7_candidate_contract": _descriptor(
            input_paths["v6_7_contract"]
        ),
        "frozen_model_binding": _descriptor(input_paths["binding"]),
        "historical_gate_contract": _descriptor(
            input_paths["historical_gate_contract"]
        ),
        "historical_replay_report": _descriptor(
            input_paths["historical_report"]
        ),
        "historical_fit_manifest": _descriptor(
            input_paths["historical_fit_manifest"]
        ),
        "collector_protocol": _descriptor(
            input_paths["collector_protocol"]
        ),
        "feature_contract": _descriptor(input_paths["feature_contract"]),
        "v8_3_profile": _descriptor(input_paths["v8_3_profile"]),
        "collector_index_snapshot": _descriptor(index_snapshot),
        **artifact_descriptors,
        "development_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in development
        ],
        "v6_2_batch_manifests": [
            _descriptor(Path(row["_manifest_path"])) for row in v6_2
        ],
        **{name: _descriptor(path) for name, path in outputs.items()},
        "final_controller_state": _descriptor(state_path),
        "parallel_target_free_freeze": _descriptor(freeze_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "parallel_freeze_sha256": report["parallel_freeze_sha256"],
        "parallel_target_free_freeze_passed": report[
            "parallel_target_free_freeze_passed"
        ],
        "future_target_access_allowed": report[
            "future_target_access_allowed"
        ],
        "decision_freeze_written_before_target_access": True,
        "outcomes_labels_settlement_returns_or_pnl_opened": False,
        "settlement_provider_called": False,
        "result_selected_extension_or_rerun_allowed": False,
        "paper_candidate_unlocked": False,
        "promotion_unlocked": False,
        "live_unlocked": False,
        "write_enabled": False,
        "wallet_enabled": False,
        "capital_at_risk": False,
    }
    manifest["manifest_id"] = canonical_payload_sha256(
        manifest,
        payload_schema_version=(
            CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION
        ),
    )
    manifest_path = run_dir / "challenge_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        report,
        report_path,
        manifest,
        manifest_path,
    )


def _one_row_per_market(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in output:
            raise ChallengeFutureFreezeError(
                f"{label} market identity missing or duplicated"
            )
        output[market_id] = row
    return output


def _required_market_row(
    rows_by_market: dict[str, dict[str, Any]],
    market_id: str,
    label: str,
) -> dict[str, Any]:
    row = rows_by_market.get(market_id)
    if row is None:
        raise ChallengeFutureFreezeError(
            f"{label} row missing for {market_id}"
        )
    return row


def _side_for_action(action: str) -> str:
    if action.startswith("BUY_UP_"):
        return "UP"
    if action.startswith("BUY_DOWN_"):
        return "DOWN"
    return "NONE"


def _relative_service_root(path: Path | str) -> str:
    resolved = Path(path).resolve()
    parts = resolved.parts
    try:
        index = parts.index("examples")
    except ValueError as exc:
        raise ChallengeFutureFreezeError(
            "service root must be under examples/"
        ) from exc
    return Path(*parts[index:]).as_posix()


__all__ = [
    "CHALLENGE_FUTURE_FREEZE_MANIFEST_SCHEMA_VERSION",
    "CHALLENGE_FUTURE_FREEZE_REPORT_SCHEMA_VERSION",
    "ChallengeFutureFreezeConfig",
    "ChallengeFutureFreezeError",
    "build_challenge_parallel_decisions",
    "build_parallel_shared_source_rows",
    "challenge_collection_status",
    "run_challenge_future_target_free_freeze",
    "select_challenge_future_window",
]
