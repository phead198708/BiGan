"""Post-collection target-free decision freeze for challenge attempt-002."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.challenge_attempt_002 import (
    BASELINE_ID,
    CANDIDATE_ID,
    NO_TRADE,
    validate_attempt_002_preregistration,
)
from bigan.v8.polymarket.challenge_attempt_002_collection import (
    COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION,
    MAXIMUM_ATTEMPTED_COUNT,
    SUPERVISOR_STATE_FILENAME,
    TARGET_QUALITY_VALID_COUNT,
    summarize_attempt_002_collection,
)
from bigan.v8.polymarket.challenge_attempt_002_pipeline import (
    build_attempt_002_target_free_pairs,
)
from bigan.v8.polymarket.challenge_future_freeze import (
    SHARED_SOURCE_ROW_SCHEMA_VERSION,
    build_parallel_shared_source_rows,
)
from bigan.v8.polymarket.challenge_historical_development import (
    SAFE_FALSES,
)
from bigan.v8.polymarket.challenge_v8_1_entry_price_floor_sizing import (
    materialize_entry_price_floor_sizing_decisions,
    validate_entry_price_floor_sizing_profile,
)
from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.exact_model_runtime_binding import (
    ExactModelRuntimeBindingConfig,
    verify_exact_model_runtime_binding,
)
from bigan.v8.polymarket.training import (
    execution_layer_v2_adaptive_support_controller_v8_1 as v81,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_canary import (
    _score_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_adaptive_support_controller_v8_1_future_holdout_pipeline import (
    _baseline_guard_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_market_clustered_mean_ev_v6_2_future_batch_canary import (
    MarketClusteredMeanEVV62FutureBatchCanaryConfig,
    run_market_clustered_mean_ev_v6_2_future_batch_canary,
)
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
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

TARGET_FREEZE_REPORT_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-target-freeze-report-v1"
)
TARGET_FREEZE_MANIFEST_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-target-freeze-manifest-v1"
)
CANDIDATE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-candidate-decision-v1"
)
BASELINE_DECISION_SCHEMA_VERSION = (
    "bigan-v8-challenge-attempt-002-baseline-decision-v1"
)


class ChallengeAttempt002TargetFreezeError(ValueError):
    """Raised when the future decisions cannot be frozen outcome-blind."""


@dataclass(frozen=True, slots=True)
class Attempt002TargetFreezeConfig:
    """Hash-pinned inputs for the post-collection target-free freeze."""

    run_id: str
    output_dir: Path | str
    service_root: Path | str
    protocol_path: Path | str
    expected_protocol_sha256: str
    supervisor_state_path: Path | str
    expected_supervisor_state_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    feature_contract_path: Path | str
    expected_feature_contract_sha256: str
    v6_2_candidate_manifest_path: Path | str
    expected_v6_2_candidate_manifest_sha256: str
    historical_fit_manifest_path: Path | str
    expected_historical_fit_manifest_sha256: str
    frozen_model_binding_path: Path | str
    expected_frozen_model_binding_sha256: str
    v8_1_candidate_contract_path: Path | str
    expected_v8_1_candidate_contract_sha256: str
    entry_price_floor_profile_path: Path | str
    expected_entry_price_floor_profile_sha256: str
    sizing_profile_path: Path | str
    expected_sizing_profile_sha256: str
    implementation_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if not _is_git_commit(self.implementation_commit):
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.decision_freeze_created_ts <= 0:
            raise ValueError("decision_freeze_created_ts must be positive")
        for name in (
            "output_dir",
            "service_root",
            "protocol_path",
            "supervisor_state_path",
            "collector_index_path",
            "feature_contract_path",
            "v6_2_candidate_manifest_path",
            "historical_fit_manifest_path",
            "frozen_model_binding_path",
            "v8_1_candidate_contract_path",
            "entry_price_floor_profile_path",
            "sizing_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "expected_protocol_sha256",
            "expected_supervisor_state_sha256",
            "expected_collector_index_sha256",
            "expected_feature_contract_sha256",
            "expected_v6_2_candidate_manifest_sha256",
            "expected_historical_fit_manifest_sha256",
            "expected_frozen_model_binding_sha256",
            "expected_v8_1_candidate_contract_sha256",
            "expected_entry_price_floor_profile_sha256",
            "expected_sizing_profile_sha256",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 digest")


def build_attempt_002_decision_freeze(
    *,
    selected_index_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    native_decisions: Sequence[Mapping[str, Any]],
    candidate_guard_rows: Sequence[Mapping[str, Any]],
    baseline_guard_rows: Sequence[Mapping[str, Any]],
    entry_price_floor_profile: Mapping[str, Any],
    sizing_profile: Mapping[str, Any],
    protocol: Mapping[str, Any],
    decision_freeze_created_ts: int,
) -> dict[str, Any]:
    """Bind the frozen v8.1 overlay and v6.7 decisions before target access."""

    validate_attempt_002_preregistration(protocol)
    validate_entry_price_floor_sizing_profile(sizing_profile)
    expected_count = int(
        protocol["future_window"]["exact_quality_valid_market_count"]
    )
    if expected_count != TARGET_QUALITY_VALID_COUNT:
        raise ChallengeAttempt002TargetFreezeError(
            "attempt-002 protocol is not exact-120"
        )
    if not all(
        len(rows) == expected_count
        for rows in (
            selected_index_rows,
            native_decisions,
            candidate_guard_rows,
            baseline_guard_rows,
        )
    ):
        raise ChallengeAttempt002TargetFreezeError(
            "selected, native, candidate, and baseline rows must be exact-120"
        )
    selected = [dict(row) for row in selected_index_rows]
    market_ids = [str(row.get("market_id") or "") for row in selected]
    if (
        "" in market_ids
        or len(market_ids) != len(set(market_ids))
        or market_ids
        != [str(row.get("market_id") or "") for row in native_decisions]
        or market_ids
        != [str(row.get("market_id") or "") for row in candidate_guard_rows]
        or market_ids
        != [str(row.get("market_id") or "") for row in baseline_guard_rows]
    ):
        raise ChallengeAttempt002TargetFreezeError(
            "target-free decision rows do not match the exact market order"
        )
    boundary = int(
        protocol["future_window"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    prior_start = boundary
    for index, row in enumerate(selected):
        start = _positive_integer(
            row.get("market_start_ts"),
            field=f"selected row {index} market_start_ts",
        )
        end = _positive_integer(
            row.get("market_end_ts"),
            field=f"selected row {index} market_end_ts",
        )
        if (
            start <= prior_start
            or end <= start
            or row.get("capture_quality_valid") is not True
            or row.get("labels_outcomes_or_pnl_opened") is not False
        ):
            raise ChallengeAttempt002TargetFreezeError(
                f"selected row {index} is not future-freeze eligible"
            )
        prior_start = start
    latest_end = max(int(row["market_end_ts"]) for row in selected)
    if decision_freeze_created_ts <= latest_end:
        raise ChallengeAttempt002TargetFreezeError(
            "decision freeze timestamp must follow the complete raw window"
        )
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_INFERENCE_FIELDS))
        | set(
            _find_nonempty_fields(feature_rows, FORBIDDEN_INFERENCE_FIELDS)
        )
        | set(
            _find_nonempty_fields(
                native_decisions,
                FORBIDDEN_INFERENCE_FIELDS,
            )
        )
        | set(
            _find_nonempty_fields(
                candidate_guard_rows,
                FORBIDDEN_INFERENCE_FIELDS,
            )
        )
        | set(
            _find_nonempty_fields(
                baseline_guard_rows,
                FORBIDDEN_INFERENCE_FIELDS,
            )
        )
    )
    if forbidden:
        raise ChallengeAttempt002TargetFreezeError(
            "target-free inputs contain forbidden fields: "
            + ",".join(forbidden)
        )

    development_decisions = materialize_entry_price_floor_sizing_decisions(
        base_guard_rows=candidate_guard_rows,
        five_action_rows=action_rows,
        frozen_market_ids=market_ids,
        profile=sizing_profile,
        entry_price_floor_profile=entry_price_floor_profile,
    )
    candidate_decisions = _future_candidate_decisions(
        development_decisions,
        protocol=protocol,
    )
    baseline_decisions = _future_baseline_decisions(
        baseline_guard_rows,
        protocol=protocol,
    )
    shared_source_rows = build_parallel_shared_source_rows(
        selected,
        baseline_guard_rows=[dict(row) for row in baseline_guard_rows],
    )
    if any(
        row.get("schema_version") != SHARED_SOURCE_ROW_SCHEMA_VERSION
        or int(row.get("policy_grid_decision_ts") or 0) <= 0
        for row in shared_source_rows
    ):
        raise ChallengeAttempt002TargetFreezeError(
            "shared source grid lacks a target-free policy timestamp"
        )
    pairs = build_attempt_002_target_free_pairs(
        shared_source_rows=shared_source_rows,
        candidate_decisions=candidate_decisions,
        baseline_decisions=baseline_decisions,
        protocol=protocol,
    )
    return {
        "market_ids": market_ids,
        "shared_source_rows": shared_source_rows,
        "candidate_decisions": candidate_decisions,
        "baseline_decisions": baseline_decisions,
        "target_free_pairs": pairs,
        "candidate_accepted_market_count": sum(
            row["selected_action"] != NO_TRADE
            for row in candidate_decisions
        ),
        "baseline_accepted_market_count": sum(
            row["selected_action"] != NO_TRADE
            for row in baseline_decisions
        ),
        "outcomes_resolution_labels_or_pnl_opened": False,
        "target_access_claim_written": False,
        "safety": SAFE_FALSES,
    }


def run_attempt_002_target_freeze(
    config: Attempt002TargetFreezeConfig,
) -> dict[str, Any]:
    """Score the completed raw window and write an immutable target-free freeze."""

    paths = {
        "protocol": config.protocol_path.resolve(),
        "supervisor_state": config.supervisor_state_path.resolve(),
        "collector_index": config.collector_index_path.resolve(),
        "feature_contract": config.feature_contract_path.resolve(),
        "v6_2_candidate_manifest": (
            config.v6_2_candidate_manifest_path.resolve()
        ),
        "historical_fit_manifest": (
            config.historical_fit_manifest_path.resolve()
        ),
        "frozen_model_binding": (
            config.frozen_model_binding_path.resolve()
        ),
        "v8_1_candidate_contract": (
            config.v8_1_candidate_contract_path.resolve()
        ),
        "entry_price_floor_profile": (
            config.entry_price_floor_profile_path.resolve()
        ),
        "sizing_profile": config.sizing_profile_path.resolve(),
    }
    expected = {
        "protocol": config.expected_protocol_sha256,
        "supervisor_state": config.expected_supervisor_state_sha256,
        "collector_index": config.expected_collector_index_sha256,
        "feature_contract": config.expected_feature_contract_sha256,
        "v6_2_candidate_manifest": (
            config.expected_v6_2_candidate_manifest_sha256
        ),
        "historical_fit_manifest": (
            config.expected_historical_fit_manifest_sha256
        ),
        "frozen_model_binding": (
            config.expected_frozen_model_binding_sha256
        ),
        "v8_1_candidate_contract": (
            config.expected_v8_1_candidate_contract_sha256
        ),
        "entry_price_floor_profile": (
            config.expected_entry_price_floor_profile_sha256
        ),
        "sizing_profile": config.expected_sizing_profile_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, expected[name], f"attempt-002 {name}")

    protocol = _load_json(paths["protocol"])
    validate_attempt_002_preregistration(protocol)
    service_root = config.service_root.resolve()
    expected_service_root = (
        config.protocol_path.resolve().parents[3]
        / str(protocol["future_window"]["service_root"])
    ).resolve()
    if (
        service_root != expected_service_root
        or paths["supervisor_state"]
        != service_root / SUPERVISOR_STATE_FILENAME
        or paths["collector_index"]
        != service_root / "persistent_outcome_blind_round_index.jsonl"
    ):
        raise ChallengeAttempt002TargetFreezeError(
            "service root, state, or collector index escaped the frozen path"
        )
    state = _load_json(paths["supervisor_state"])
    _validate_completed_collection_state(
        state,
        protocol=protocol,
        service_root=service_root,
        protocol_sha256=expected["protocol"],
    )
    source_index_sha256 = _sha256_file(paths["collector_index"])
    index_rows = load_and_validate_persistent_outcome_blind_index(
        paths["collector_index"]
    )
    selected, selection = _select_exact_attempt_002_rows(
        index_rows,
        protocol=protocol,
    )
    if (
        selection["target_reached"] is not True
        or selection["quality_valid_market_count"]
        != TARGET_QUALITY_VALID_COUNT
        or selection["selected_market_ids"]
        != [str(row["market_id"]) for row in selected]
    ):
        raise ChallengeAttempt002TargetFreezeError(
            "collector index does not contain the frozen exact-120 window"
        )
    for row in selected:
        _verify_index_raw_descriptors(row)

    artifacts = _load_exact_model_artifacts(
        fit_manifest_path=paths["historical_fit_manifest"],
        binding_path=paths["frozen_model_binding"],
        candidate_contract_path=paths["v8_1_candidate_contract"],
        fit_manifest_sha256=expected["historical_fit_manifest"],
        binding_sha256=expected["frozen_model_binding"],
        candidate_contract_sha256=expected["v8_1_candidate_contract"],
    )
    entry_profile = _load_json(paths["entry_price_floor_profile"])
    sizing_profile = _load_json(paths["sizing_profile"])
    validate_entry_price_floor_sizing_profile(sizing_profile)

    development = _discover_development_manifests(
        service_root=service_root,
        selected_index_rows=selected,
        feature_contract_sha256=expected["feature_contract"],
    )
    run_dir = _prepare_run_dir(
        config.output_dir,
        config.run_id,
        overwrite=config.overwrite_existing,
    )
    v6_2 = _score_v6_2_batches(
        development=development,
        output_dir=run_dir / "v6_2_batch_canary_runs",
        candidate_manifest_path=paths["v6_2_candidate_manifest"],
        candidate_manifest_sha256=expected["v6_2_candidate_manifest"],
    )
    action_rows, feature_rows, scored_rows = _load_target_free_rows(
        development,
        v6_2,
        selected_market_ids={str(row["market_id"]) for row in selected},
    )
    selected_ids = [str(row["market_id"]) for row in selected]
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
    native_decisions, candidate_guard, final_state = _score_window(
        selected_ids,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=artifacts["model"],
        v6_7_profile=artifacts["v6_7_profile"],
    )
    baseline_guard = _baseline_guard_window(
        selected_ids,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        v6_7_profile=artifacts["v6_7_profile"],
    )
    freeze = build_attempt_002_decision_freeze(
        selected_index_rows=selected,
        action_rows=action_rows,
        feature_rows=feature_rows,
        native_decisions=native_decisions,
        candidate_guard_rows=candidate_guard,
        baseline_guard_rows=baseline_guard,
        entry_price_floor_profile=entry_profile,
        sizing_profile=sizing_profile,
        protocol=protocol,
        decision_freeze_created_ts=config.decision_freeze_created_ts,
    )
    if _sha256_file(paths["collector_index"]) != source_index_sha256:
        raise ChallengeAttempt002TargetFreezeError(
            "collector index changed while decisions were frozen"
        )
    return _write_freeze(
        run_dir=run_dir,
        config=config,
        paths=paths,
        protocol=protocol,
        state=state,
        selection=selection,
        source_index_sha256=source_index_sha256,
        selected=selected,
        action_rows=action_rows,
        feature_rows=feature_rows,
        scored_rows=scored_rows,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        native_decisions=native_decisions,
        candidate_guard=candidate_guard,
        baseline_guard=baseline_guard,
        final_state=final_state,
        freeze=freeze,
        development=development,
        v6_2=v6_2,
        model_artifacts=artifacts,
        v6_7_candidate_summary=v6_7_candidate_summary,
        canonical_summary=canonical_summary,
    )


def _select_exact_attempt_002_rows(
    index_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    boundary = int(
        protocol["future_window"][
            "strictly_later_minimum_market_start_ts_exclusive"
        ]
    )
    summary = summarize_attempt_002_collection(
        index_rows,
        boundary_exclusive=boundary,
    )
    selected_ids = list(summary["selected_market_ids"])
    selected_set = set(selected_ids)
    ordered = sorted(
        (dict(row) for row in index_rows),
        key=lambda row: (
            int(row.get("scheduled_round_start_ts") or 0),
            int(row.get("sequence") or 0),
        ),
    )[:MAXIMUM_ATTEMPTED_COUNT]
    by_id = {
        str(row.get("market_id") or ""): row
        for row in ordered
        if str(row.get("market_id") or "") in selected_set
    }
    if set(by_id) != selected_set:
        raise ChallengeAttempt002TargetFreezeError(
            "selected market identity cannot be reconstructed"
        )
    return [by_id[market_id] for market_id in selected_ids], summary


def _validate_completed_collection_state(
    state: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    service_root: Path,
    protocol_sha256: str,
) -> None:
    checks = {
        "schema": state.get("schema_version")
        == COLLECTION_SUPERVISOR_STATE_SCHEMA_VERSION,
        "attempt": state.get("attempt_id") == protocol["attempt_id"],
        "protocol": state.get("protocol_sha256")
        == protocol_sha256.lower(),
        "service_root": state.get("service_root") == str(service_root),
        "status": state.get("status") == "quality_valid_target_reached"
        and state.get("collection_complete") is True
        and state.get("fail_closed") is False,
        "count": state.get("quality_valid_market_count")
        == TARGET_QUALITY_VALID_COUNT
        and 120 <= int(state.get("attempted_market_count") or 0) <= 180,
        "collector_stopped": state.get("collector_pid") is None,
        "outcomes": state.get(
            "outcomes_resolution_labels_or_pnl_opened"
        )
        is False,
        "safety": state.get("safety") == SAFE_FALSES,
    }
    _raise_failed_checks("completed collection state", checks)


def _discover_development_manifests(
    *,
    service_root: Path,
    selected_index_rows: Sequence[Mapping[str, Any]],
    feature_contract_sha256: str,
) -> list[dict[str, Any]]:
    batch_ids = {
        str(row.get("batch_id") or "") for row in selected_index_rows
    }
    if "" in batch_ids:
        raise ChallengeAttempt002TargetFreezeError(
            "selected collector batch id missing"
        )
    by_batch: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(
        service_root.glob(
            "batch_canary_runs/*/"
            "execution_layer_v2_outcome_blind_batch_canary_manifest.json"
        )
    ):
        manifest = _load_json(path)
        batch_id = str(manifest.get("batch_id") or "")
        if not batch_id or batch_id in by_batch:
            raise ChallengeAttempt002TargetFreezeError(
                "development batch manifest identity missing or duplicated"
            )
        by_batch[batch_id] = (path.resolve(), manifest)
    output = []
    for batch_id in sorted(batch_ids):
        pair = by_batch.get(batch_id)
        if pair is None:
            raise ChallengeAttempt002TargetFreezeError(
                f"development batch manifest missing for {batch_id}"
            )
        path, manifest = pair
        feature = _verified_descriptor(
            manifest.get("feature_contract"),
            "attempt-002 batch feature contract",
        )
        checks = {
            "development_passed": manifest.get(
                "development_data_canary_passed"
            )
            is True,
            "no_candidate_scoring": manifest.get(
                "candidate_model_scoring_attempted"
            )
            is False,
            "no_target_access": manifest.get(
                "labels_outcomes_or_pnl_opened"
            )
            is False,
            "feature_contract": feature["sha256"]
            == feature_contract_sha256.lower(),
            "collection_binding": manifest.get(
                "exact_model_runtime_binding_required"
            )
            is False
            and manifest.get("exact_model_runtime_binding_verified")
            is False
            and manifest.get("exact_model_runtime_binding_summary") is None,
        }
        _raise_failed_checks(f"development batch {batch_id}", checks)
        manifest["_manifest_path"] = str(path)
        output.append(manifest)
    return output


def _score_v6_2_batches(
    *,
    development: Sequence[Mapping[str, Any]],
    output_dir: Path,
    candidate_manifest_path: Path,
    candidate_manifest_sha256: str,
) -> list[dict[str, Any]]:
    output = []
    for sequence, manifest in enumerate(development, start=1):
        dev_path = Path(str(manifest["_manifest_path"])).resolve()
        result = run_market_clustered_mean_ev_v6_2_future_batch_canary(
            MarketClusteredMeanEVV62FutureBatchCanaryConfig(
                run_id=(
                    f"attempt-002-postcollection-v6-2-{sequence:03d}-"
                    f"{manifest['batch_id']}"
                ),
                output_dir=output_dir,
                development_batch_canary_manifest_path=dev_path,
                expected_development_batch_canary_manifest_sha256=(
                    _sha256_file(dev_path)
                ),
                candidate_manifest_path=candidate_manifest_path,
                expected_candidate_manifest_sha256=(
                    candidate_manifest_sha256
                ),
            )
        )
        score_manifest = _load_json(
            Path(str(result["manifest_path"])).resolve()
        )
        score_manifest["_manifest_path"] = result["manifest_path"]
        output.append(score_manifest)
    return output


def _load_target_free_rows(
    development: Sequence[Mapping[str, Any]],
    v6_2: Sequence[Mapping[str, Any]],
    *,
    selected_market_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        dev_path = Path(str(dev["_manifest_path"])).resolve()
        matched = _verified_descriptor(
            score.get("development_batch_canary_manifest"),
            "attempt-002 matched development manifest",
        )
        if matched["sha256"] != _sha256_file(dev_path):
            raise ChallengeAttempt002TargetFreezeError(
                "v6.2 batch does not match its development manifest"
            )
        actions.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev.get("five_action_grid"),
                        "attempt-002 five-action grid",
                    )["path"]
                )
            )
        )
        features.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        dev.get("feature_rows"),
                        "attempt-002 feature rows",
                    )["path"]
                )
            )
        )
        scored.extend(
            _load_jsonl(
                Path(
                    _verified_descriptor(
                        score.get("mean_ev_scored_rows"),
                        "attempt-002 v6.2 scored rows",
                    )["path"]
                )
            )
        )
    filtered = tuple(
        [
            row
            for row in rows
            if str(row.get("market_id") or "") in selected_market_ids
        ]
        for rows in (actions, features, scored)
    )
    for name, rows in zip(
        ("action", "feature", "scored"),
        filtered,
        strict=True,
    ):
        if {str(row.get("market_id") or "") for row in rows} != (
            selected_market_ids
        ):
            raise ChallengeAttempt002TargetFreezeError(
                f"{name} rows do not cover the exact-120 market grid"
            )
    return filtered


def _load_exact_model_artifacts(
    *,
    fit_manifest_path: Path,
    binding_path: Path,
    candidate_contract_path: Path,
    fit_manifest_sha256: str,
    binding_sha256: str,
    candidate_contract_sha256: str,
) -> dict[str, Any]:
    fit = _load_json(fit_manifest_path)
    binding = _load_json(binding_path)
    contract = _load_json(candidate_contract_path)
    model_descriptor = _verified_descriptor(
        fit.get("model"),
        "attempt-002 frozen v8.1 model",
    )
    profile_descriptor = _verified_descriptor(
        fit.get("profile"),
        "attempt-002 frozen v8.1 profile",
    )
    v6_7_descriptor = _verified_descriptor(
        fit.get("v6_7_candidate_profile"),
        "attempt-002 frozen v6.7 profile",
    )
    v7_0_descriptor = _verified_descriptor(
        fit.get("v7_0_training_profile"),
        "attempt-002 frozen v7.0 profile",
    )
    training_descriptor = _verified_descriptor(
        fit.get("seed_runtime_target_rows"),
        "attempt-002 source training rows",
    )
    runtime_summary = verify_exact_model_runtime_binding(
        ExactModelRuntimeBindingConfig(
            candidate_contract_path=candidate_contract_path,
            expected_candidate_contract_sha256=candidate_contract_sha256,
            frozen_model_binding_path=binding_path,
            expected_frozen_model_binding_sha256=binding_sha256,
            frozen_model_artifact_path=Path(model_descriptor["path"]),
            expected_frozen_model_artifact_sha256=model_descriptor[
                "sha256"
            ],
            candidate_profile_path=Path(profile_descriptor["path"]),
            expected_candidate_profile_sha256=profile_descriptor["sha256"],
        )
    )
    checks = {
        "fit_manifest": binding.get("historical_fit_manifest_sha256")
        == fit_manifest_sha256.lower(),
        "model": binding.get("frozen_model_artifact_sha256")
        == model_descriptor["sha256"],
        "profile": binding.get("frozen_profile_sha256")
        == profile_descriptor["sha256"]
        == contract.get("profile_sha256"),
        "training": binding.get("source_training_rows_sha256")
        == training_descriptor["sha256"]
        == contract.get("source_model_hash"),
        "v6_7": v6_7_descriptor["sha256"]
        == contract.get("fallback_profile_sha256", v6_7_descriptor["sha256"]),
        "frozen": fit.get("target_free_canary_collection_allowed") is True
        and fit.get("promotion_evidence_eligible") is False
        and fit.get("paper_candidate_allowed") is False
        and fit.get("live_trading_enabled") is False
        and fit.get("polymarket_write_enabled") is False
        and fit.get("wallet_signing_enabled") is False
        and fit.get("capital_at_risk") is False,
    }
    _raise_failed_checks("exact v8.1 model artifacts", checks)
    model = _load_json(Path(model_descriptor["path"]))
    v8_1_profile = _load_json(Path(profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    v81.validate_adaptive_support_controller_v8_1_profile(v8_1_profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)
    return {
        "model": model,
        "v8_1_profile": v8_1_profile,
        "v6_7_profile": v6_7_profile,
        "v7_0_profile": v7_0_profile,
        "runtime_binding_summary": runtime_summary,
        "descriptors": {
            "model": model_descriptor,
            "v8_1_profile": profile_descriptor,
            "v6_7_profile": v6_7_descriptor,
            "v7_0_profile": v7_0_descriptor,
            "source_training_rows": training_descriptor,
        },
    }


def _future_candidate_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in decisions:
        payload = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "decision_id",
                "historical_development_only",
                "promotion_evidence_eligible",
                "schema_version",
            }
        }
        payload.update(
            {
                "schema_version": CANDIDATE_DECISION_SCHEMA_VERSION,
                "attempt_id": protocol["attempt_id"],
                "candidate_id": CANDIDATE_ID,
                "historical_development_data_used": False,
                "future_attempt_002_target_free": True,
                "outcomes_resolution_labels_or_pnl_opened": False,
                "promotion_evidence_eligible_before_future_gate": False,
                "safety": SAFE_FALSES,
            }
        )
        payload["decision_id"] = canonical_json_sha256(payload)
        output.append(payload)
    return output


def _future_baseline_decisions(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        action = str(row.get("selected_action") or "")
        side = str(row.get("selected_side") or "")
        selected = action != NO_TRADE
        payload = {
            "schema_version": BASELINE_DECISION_SCHEMA_VERSION,
            "attempt_id": protocol["attempt_id"],
            "candidate_id": BASELINE_ID,
            "market_id": str(row.get("market_id") or ""),
            "decision_ts": _positive_integer(
                row.get("decision_ts"),
                field="baseline decision_ts",
            ),
            "selected_action": action,
            "selected_side": side,
            "execution_guard_order_allowed": row.get(
                "execution_guard_order_allowed"
            )
            is True,
            "execution_blocking_reason_codes": sorted(
                str(value)
                for value in (
                    row.get("execution_blocking_reason_codes") or []
                )
            ),
            "baseline_fixed_position_size": 0.2,
            "baseline_position_size": 0.2 if selected else 0.0,
            "historical_development_data_used": False,
            "target_used_as_decision_time_input": False,
            "outcome_or_pnl_field_used_at_inference": False,
            "outcomes_resolution_labels_or_pnl_opened": False,
            "future_attempt_002_target_free": True,
            "safety": SAFE_FALSES,
        }
        if (
            not payload["market_id"]
            or _side(action) != side
            or (selected and not payload["execution_guard_order_allowed"])
        ):
            raise ChallengeAttempt002TargetFreezeError(
                "baseline guard decision is invalid"
            )
        payload["decision_id"] = canonical_json_sha256(payload)
        output.append(payload)
    return output


def _write_freeze(
    *,
    run_dir: Path,
    config: Attempt002TargetFreezeConfig,
    paths: Mapping[str, Path],
    protocol: Mapping[str, Any],
    state: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_index_sha256: str,
    selected: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    scored_rows: Sequence[Mapping[str, Any]],
    canonical_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    native_decisions: Sequence[Mapping[str, Any]],
    candidate_guard: Sequence[Mapping[str, Any]],
    baseline_guard: Sequence[Mapping[str, Any]],
    final_state: Mapping[str, Any],
    freeze: Mapping[str, Any],
    development: Sequence[Mapping[str, Any]],
    v6_2: Sequence[Mapping[str, Any]],
    model_artifacts: Mapping[str, Any],
    v6_7_candidate_summary: Mapping[str, Any],
    canonical_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output_paths = {
        "selected_index_rows": run_dir
        / "attempt_002_selected_index_rows.jsonl",
        "action_rows": run_dir / "attempt_002_five_action_rows.jsonl",
        "feature_rows": run_dir / "attempt_002_feature_rows.jsonl",
        "scored_rows": run_dir / "attempt_002_v6_2_scored_rows.jsonl",
        "canonical_rows": run_dir / "attempt_002_canonical_rows.jsonl",
        "baseline_rows": run_dir / "attempt_002_v6_7_baseline_rows.jsonl",
        "native_decisions": run_dir
        / "attempt_002_v8_1_native_decisions.jsonl",
        "candidate_guard": run_dir
        / "attempt_002_v8_1_guard_decisions.jsonl",
        "baseline_guard": run_dir
        / "attempt_002_v6_7_guard_decisions.jsonl",
        "shared_source_rows": run_dir
        / "attempt_002_shared_source_rows.jsonl",
        "candidate_decisions": run_dir
        / "attempt_002_candidate_decisions.jsonl",
        "baseline_decisions": run_dir
        / "attempt_002_baseline_decisions.jsonl",
        "target_free_pairs": run_dir
        / "attempt_002_target_free_pairs.jsonl",
    }
    rows_by_name = {
        "selected_index_rows": selected,
        "action_rows": action_rows,
        "feature_rows": feature_rows,
        "scored_rows": scored_rows,
        "canonical_rows": canonical_rows,
        "baseline_rows": baseline_rows,
        "native_decisions": native_decisions,
        "candidate_guard": candidate_guard,
        "baseline_guard": baseline_guard,
        "shared_source_rows": freeze["shared_source_rows"],
        "candidate_decisions": freeze["candidate_decisions"],
        "baseline_decisions": freeze["baseline_decisions"],
        "target_free_pairs": freeze["target_free_pairs"],
    }
    for name, rows in rows_by_name.items():
        _write_jsonl(output_paths[name], list(rows))
    index_snapshot = run_dir / "attempt_002_collector_index_snapshot.jsonl"
    shutil.copyfile(paths["collector_index"], index_snapshot)
    if _sha256_file(index_snapshot) != source_index_sha256:
        raise ChallengeAttempt002TargetFreezeError(
            "collector index snapshot hash mismatch"
        )
    final_state_path = run_dir / "attempt_002_final_controller_state.json"
    _write_json(final_state_path, dict(final_state))
    report = {
        "schema_version": TARGET_FREEZE_REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "attempt_id": protocol["attempt_id"],
        "model_version": protocol["model_version"],
        "candidate_id": CANDIDATE_ID,
        "baseline_id": BASELINE_ID,
        "implementation_commit": config.implementation_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "collector_index_sha256": source_index_sha256,
        "supervisor_state_sha256": _sha256_file(
            paths["supervisor_state"]
        ),
        "collection_status": state["status"],
        "attempted_market_count": selection["attempted_market_count"],
        "exact_quality_valid_market_count": len(selected),
        "candidate_accepted_market_count": freeze[
            "candidate_accepted_market_count"
        ],
        "baseline_accepted_market_count": freeze[
            "baseline_accepted_market_count"
        ],
        "v6_7_candidate_summary": dict(v6_7_candidate_summary),
        "canonical_summary": dict(canonical_summary),
        "runtime_binding_summary": dict(
            model_artifacts["runtime_binding_summary"]
        ),
        "all_decisions_frozen_before_target_access": True,
        "target_access_claim_written": False,
        "candidate_scoring_during_raw_capture": False,
        "postcollection_target_free_scoring_performed": True,
        "settlement_or_resolution_provider_called": False,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "collection_control_invoked": False,
        "safety": SAFE_FALSES,
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "attempt_002_target_freeze_report.json"
    report_md_path = run_dir / "attempt_002_target_freeze_report.md"
    _write_json(report_path, report)
    _write_text(
        report_md_path,
        "\n".join(
            [
                "# Attempt-002 Target-Free Decision Freeze",
                "",
                f"- exact future markets: `{len(selected)}`",
                (
                    "- candidate / baseline accepted: "
                    f"`{freeze['candidate_accepted_market_count']} / "
                    f"{freeze['baseline_accepted_market_count']}`"
                ),
                f"- collector index: `{source_index_sha256}`",
                "- target-access claim written: `false`",
                "- outcomes, labels, settlement, and PnL opened: `false`",
                "- collection control invoked: `false`",
                "- paper/live/write/wallet/handoff/promotion remain false.",
                "",
            ]
        ),
    )
    manifest = {
        "schema_version": TARGET_FREEZE_MANIFEST_SCHEMA_VERSION,
        "run_id": config.run_id,
        "attempt_id": protocol["attempt_id"],
        "implementation_commit": config.implementation_commit,
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "protocol": _descriptor(paths["protocol"]),
        "collection_supervisor_state": _descriptor(
            paths["supervisor_state"]
        ),
        "collector_index_snapshot": _descriptor(index_snapshot),
        "feature_contract": _descriptor(paths["feature_contract"]),
        "v6_2_candidate_manifest": _descriptor(
            paths["v6_2_candidate_manifest"]
        ),
        "historical_fit_manifest": _descriptor(
            paths["historical_fit_manifest"]
        ),
        "frozen_model_binding": _descriptor(
            paths["frozen_model_binding"]
        ),
        "v8_1_candidate_contract": _descriptor(
            paths["v8_1_candidate_contract"]
        ),
        "entry_price_floor_profile": _descriptor(
            paths["entry_price_floor_profile"]
        ),
        "sizing_profile": _descriptor(paths["sizing_profile"]),
        **dict(model_artifacts["descriptors"]),
        "development_batch_manifests": [
            _descriptor(Path(str(row["_manifest_path"])))
            for row in development
        ],
        "v6_2_batch_manifests": [
            _descriptor(Path(str(row["_manifest_path"]))) for row in v6_2
        ],
        **{
            name: _descriptor(path)
            for name, path in output_paths.items()
        },
        "final_controller_state": _descriptor(final_state_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "exact_market_count": len(selected),
        "target_free_pairs_sha256": _sha256_file(
            output_paths["target_free_pairs"]
        ),
        "all_decisions_frozen_before_target_access": True,
        "target_access_claim_written": False,
        "outcomes_resolution_labels_or_pnl_opened": False,
        "collection_control_invoked": False,
        "safety": SAFE_FALSES,
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "attempt_002_target_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        report,
        report_path,
        manifest,
        manifest_path,
    )


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ChallengeAttempt002TargetFreezeError(
            f"{field} is not an integer"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ChallengeAttempt002TargetFreezeError(
            f"{field} is not an integer"
        ) from error
    if number <= 0 or value != number:
        raise ChallengeAttempt002TargetFreezeError(
            f"{field} must be a positive integer"
        )
    return number


def _side(action: str) -> str:
    if action == NO_TRADE:
        return "NONE"
    if action.startswith("BUY_UP_"):
        return "UP"
    if action.startswith("BUY_DOWN_"):
        return "DOWN"
    return ""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value.lower()
        )
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(
            character in "0123456789abcdef"
            for character in value.lower()
        )
    )


def _raise_failed_checks(
    label: str,
    checks: Mapping[str, bool],
) -> None:
    blockers = [name for name, passed in checks.items() if not passed]
    if blockers:
        raise ChallengeAttempt002TargetFreezeError(
            f"{label} invalid: {','.join(blockers)}"
        )


__all__ = [
    "Attempt002TargetFreezeConfig",
    "BASELINE_DECISION_SCHEMA_VERSION",
    "CANDIDATE_DECISION_SCHEMA_VERSION",
    "ChallengeAttempt002TargetFreezeError",
    "TARGET_FREEZE_MANIFEST_SCHEMA_VERSION",
    "TARGET_FREEZE_REPORT_SCHEMA_VERSION",
    "build_attempt_002_decision_freeze",
    "run_attempt_002_target_freeze",
]
