"""Target-free freeze and single-use confirmatory gate for issue #231 v6.9."""

from __future__ import annotations

import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9 import (
    CANDIDATE_NAME,
    FORBIDDEN_TARGET_FIELDS,
    apply_v6_9_score_to_runtime_pnl_mapping,
)
from bigan.v8.polymarket.training.execution_layer_v2_calibration_scale_aligned_runtime_pnl_v6_9_future_batch_canary import (
    _prior_market_reference,
    _validate_candidate_manifest,
    validate_v6_9_future_collection_plan,
)
from bigan.v8.polymarket.training.execution_layer_v2_conformal_v5_future_settlement import (
    _finalize_selected_rounds,
    _is_retryable_settlement_failure,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_pipeline import (
    _validate_outcome_blind_index_row,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7_post_freeze import (
    _legacy_guard_accepted_sbc_decisions,
    _runtime_targets_for_decisions,
)
from bigan.v8.polymarket.training.execution_layer_v2_persistent_outcome_blind_collector import (
    load_and_validate_persistent_outcome_blind_index,
)
from bigan.v8.polymarket.training.execution_layer_v2_policy_selected_conformal_net_return_v6 import (
    _find_nonempty_fields,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8 import (
    build_regime_emergent_target_free_support,
    build_v6_8_regime_emergent_confirmatory_gate,
    validate_regime_emergent_pnl_v6_8_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_confirmatory import (
    FROZEN_RUNTIME_POLICY_PROFILE_SHA256,
)
from bigan.v8.polymarket.training.execution_layer_v2_regime_emergent_pnl_v6_8_pipeline import (
    FROZEN_EVALUATION_PROFILE_SHA256,
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

SCHEMA_PREFIX = "bigan-v8-calibration-scale-aligned-v6-9-confirmatory"
FREEZE_SCHEMA_VERSION = f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1"
WINDOW_MARKET_COUNT = 120
MINIMUM_GUARD_ACCEPTED_MARKET_COUNT = 40
STAGES = {"settle", "evaluate_confirmatory"}
FIVE_ACTIONS = {
    "BUY_UP_SELL_BEFORE_CLOSE",
    "BUY_DOWN_SELL_BEFORE_CLOSE",
    "BUY_UP_HOLD_TO_SETTLEMENT",
    "BUY_DOWN_HOLD_TO_SETTLEMENT",
    "NO_TRADE",
}


def _blocked_confirmatory_safety_fields() -> dict[str, bool]:
    return {**_blocked_safety_fields(), "paper_candidate_allowed": False}


@dataclass(frozen=True, slots=True)
class V69ConfirmatoryFreezeConfig:
    """Pinned inputs for the one target-free v6.9 confirmatory freeze."""

    run_id: str
    output_dir: Path | str
    service_root_path: Path | str
    candidate_manifest_path: Path | str
    expected_candidate_manifest_sha256: str
    collection_plan_path: Path | str
    expected_collection_plan_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    implementation_commit: str
    decision_freeze_created_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip() or self.decision_freeze_created_ts <= 0:
            raise ValueError("#231 confirmatory run id and freeze timestamp are required")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_candidate_manifest_sha256",
            "expected_collection_plan_sha256",
            "expected_collector_index_sha256",
            "expected_evaluation_profile_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "service_root_path",
            "candidate_manifest_path",
            "collection_plan_path",
            "collector_index_path",
            "evaluation_profile_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class V69ConfirmatoryPostFreezeConfig:
    """Pinned inputs for settlement or the single-use execution-PnL gate."""

    stage: Literal["settle", "evaluate_confirmatory"]
    run_id: str
    output_dir: Path | str
    evaluation_profile_path: Path | str
    expected_evaluation_profile_sha256: str
    prediction_freeze_manifest_path: Path | str
    expected_prediction_freeze_manifest_sha256: str
    implementation_commit: str
    stage_started_ts: int
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
        if self.stage not in STAGES or not self.run_id.strip():
            raise ValueError("#231 confirmatory stage and run_id are required")
        if self.stage_started_ts <= 0:
            raise ValueError("#231 confirmatory stage timestamp must be positive")
        _require_git_sha(self.implementation_commit)
        for name in (
            "expected_evaluation_profile_sha256",
            "expected_prediction_freeze_manifest_sha256",
        ):
            _require_sha256(str(getattr(self, name)), name=name)
        if self.stage == "evaluate_confirmatory":
            for name in (
                "runtime_policy_profile_path",
                "expected_runtime_policy_profile_sha256",
                "settled_corpus_index_path",
                "expected_settled_corpus_index_sha256",
            ):
                if getattr(self, name) in (None, ""):
                    raise ValueError(f"#231 confirmatory evaluation input missing: {name}")
            for name in (
                "expected_runtime_policy_profile_sha256",
                "expected_settled_corpus_index_sha256",
            ):
                _require_sha256(str(getattr(self, name)), name=name)
        for name in (
            "output_dir",
            "evaluation_profile_path",
            "prediction_freeze_manifest_path",
            "runtime_policy_profile_path",
            "settled_corpus_index_path",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


def select_v6_9_confirmatory_index_rows(
    index_rows: list[dict[str, Any]],
    *,
    candidate_manifest: dict[str, Any],
    collection_plan: dict[str, Any],
    candidate_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the earliest exact target under the frozen scan cap."""

    _validate_candidate_manifest(candidate_manifest)
    validate_v6_9_future_collection_plan(
        collection_plan,
        candidate_manifest=candidate_manifest,
        candidate_manifest_sha256=candidate_manifest_sha256,
    )
    prior_ids, _ = _prior_market_reference(candidate_manifest)
    boundary = int(candidate_manifest["candidate_freeze_created_ts"])
    target = int(collection_plan["target_quality_valid_market_count"])
    scan_cap = int(collection_plan["maximum_attempted_market_count"])
    eligible = [
        row
        for row in index_rows
        if int(row.get("scheduled_round_start_ts") or 0) > boundary
    ][:scan_cap]
    attempted: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    seen = set(prior_ids)
    for row in eligible:
        attempted.append(row)
        _validate_outcome_blind_index_row(row)
        quality = row.get("capture_quality_valid")
        if quality is False:
            continue
        if quality is not True:
            raise ValueError("#231 capture quality status is not explicit")
        market_id = str(row.get("market_id") or "")
        if not market_id or market_id in seen:
            raise ValueError("#231 future market identity missing or overlapping")
        selected.append(row)
        seen.add(market_id)
        if len(selected) == target:
            break
    if target != WINDOW_MARKET_COUNT:
        raise ValueError("#231 frozen confirmatory target changed")
    if len(selected) != target:
        raise ValueError("#231 insufficient quality-valid rows before scan cap")
    if not attempted:
        raise ValueError("#231 confirmatory attempted window is empty")
    return selected, attempted


def freeze_v6_9_confirmatory_window(
    config: V69ConfirmatoryFreezeConfig,
) -> dict[str, Any]:
    """Freeze earliest exact-120 decisions from pinned target-free batch artifacts."""

    inputs = _verified_freeze_inputs(config)
    candidate = _load_json(inputs["candidate_manifest"])
    plan = _load_json(inputs["collection_plan"])
    index_rows = load_and_validate_persistent_outcome_blind_index(inputs["collector_index"])
    selected, attempted = select_v6_9_confirmatory_index_rows(
        index_rows,
        candidate_manifest=candidate,
        collection_plan=plan,
        candidate_manifest_sha256=config.expected_candidate_manifest_sha256,
    )
    if config.decision_freeze_created_ts <= max(int(row["market_end_ts"]) for row in selected):
        raise ValueError("#231 confirmatory freeze attempted before all markets closed")

    materialized = _materialize_pinned_batch_artifacts(
        service_root=inputs["service_root"],
        selected_index_rows=selected,
        candidate_manifest_path=inputs["candidate_manifest"],
        collection_plan_path=inputs["collection_plan"],
    )
    selected_ids = [str(row["market_id"]) for row in selected]
    selected_id_set = set(selected_ids)
    _validate_materialized_target_free_rows(materialized, selected_ids=selected_id_set)
    mapping = _load_json(
        Path(_verified_descriptor(candidate["mapping_artifact"], "v6.9 mapping artifact")["path"])
    )
    recomputed_mapped = apply_v6_9_score_to_runtime_pnl_mapping(
        materialized["base_selected_rows"], mapping_artifact=mapping
    )
    if recomputed_mapped != materialized["mapped_rows"]:
        raise ValueError("#231 frozen mapped rows differ from pinned batch artifacts")
    recomputed_accepted = [
        row
        for row in recomputed_mapped
        if row.get("microstructure_safety_passed") is True
        and row.get("hard_execution_safety_thresholds_unchanged") is True
        and row.get("exposure_duplicate_position_and_sizing_guards_unchanged") is True
    ]
    if recomputed_accepted != materialized["accepted_rows"]:
        raise ValueError("#231 frozen guard acceptance differs from pinned batch artifacts")
    support = build_regime_emergent_target_free_support(
        materialized["accepted_rows"],
        exact_window_market_count=len(selected),
        expected_window_market_count=WINDOW_MARKET_COUNT,
        required_total_market_count=MINIMUM_GUARD_ACCEPTED_MARKET_COUNT,
        score_field="v6_9_calibrated_runtime_expected_pnl_per_contract",
    )

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    paths = {
        "selected_window_rows": run_dir / "v6_9_confirmatory_selected_index_rows.jsonl",
        "attempted_window_rows": run_dir / "v6_9_confirmatory_attempted_index_rows.jsonl",
        "target_free_feature_rows": run_dir / "v6_9_confirmatory_feature_rows.jsonl",
        "target_free_five_action_rows": run_dir / "v6_9_confirmatory_five_action_rows.jsonl",
        "v6_2_target_free_predictions": run_dir / "v6_2_confirmatory_predictions.jsonl",
        "v6_7_candidate_rows": run_dir / "v6_7_confirmatory_candidate_rows.jsonl",
        "v6_7_base_selected_rows": run_dir / "v6_7_confirmatory_base_selected_rows.jsonl",
        "v6_9_mapped_rows": run_dir / "v6_9_confirmatory_mapped_rows.jsonl",
        "v6_9_selected_decisions": run_dir / "v6_9_confirmatory_selected_decisions.jsonl",
        "matched_legacy_guard_replay": run_dir / "v6_2_confirmatory_legacy_guard_replay.jsonl",
    }
    output_rows = {
        "selected_window_rows": selected,
        "attempted_window_rows": attempted,
        "target_free_feature_rows": materialized["feature_rows"],
        "target_free_five_action_rows": materialized["action_rows"],
        "v6_2_target_free_predictions": materialized["prediction_rows"],
        "v6_7_candidate_rows": materialized["candidate_rows"],
        "v6_7_base_selected_rows": materialized["base_selected_rows"],
        "v6_9_mapped_rows": materialized["mapped_rows"],
        "v6_9_selected_decisions": materialized["accepted_rows"],
        "matched_legacy_guard_replay": materialized["legacy_replay_rows"],
    }
    for name, rows in output_rows.items():
        _write_jsonl(paths[name], rows)

    decision = {
        "schema_version": f"{SCHEMA_PREFIX}-decision-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "decision_freeze_created_ts": config.decision_freeze_created_ts,
        "selected_window_market_count": len(selected),
        "selected_window_market_ids": selected_ids,
        "attempted_index_row_count": len(attempted),
        "attempted_sequence_start": int(attempted[0]["sequence"]),
        "attempted_sequence_end": int(attempted[-1]["sequence"]),
        "target_free_support": support,
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "all_selected_markets_closed_before_freeze": True,
        "side_count_hard_gate_enabled": False,
        "side_quota_applied": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        "threshold_or_guard_tuning_performed": False,
        "manual_approval_does_not_bypass_execution_pnl_gate": True,
        **_blocked_confirmatory_safety_fields(),
    }
    decision["decision_freeze_id"] = canonical_json_sha256(decision)
    decision_path = run_dir / "v6_9_confirmatory_accepted_bet_decision_freeze.json"
    _write_json(decision_path, decision)
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-report-v1",
        "run_id": config.run_id,
        "candidate_name": CANDIDATE_NAME,
        "selected_window_market_count": len(selected),
        "attempted_index_row_count": len(attempted),
        "guard_accepted_market_count": len(materialized["accepted_rows"]),
        "guard_accepted_side_count_diagnostic": _side_distribution(
            materialized["accepted_rows"]
        ),
        "side_count_hard_gate_enabled": False,
        "target_free_support_gate_passed": support["target_free_support_gate_passed"],
        "target_free_support_blocking_reason_codes": support["blocking_reason_codes"],
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "batch_manifest_count": len(materialized["batch_manifest_paths"]),
        "complete_five_action_grid_passed": True,
        "feature_causality_violation_count": 0,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "source_score_mutated": False,
        **_blocked_confirmatory_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_9_confirmatory_target_free_freeze_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _freeze_markdown(report))
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "implementation_commit": config.implementation_commit,
        "candidate_manifest": _descriptor(inputs["candidate_manifest"]),
        "collection_plan": _descriptor(inputs["collection_plan"]),
        "collector_index": _descriptor(inputs["collector_index"]),
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "batch_action_liveness_manifests": [
            _descriptor(path) for path in materialized["batch_manifest_paths"]
        ],
        **{name: _descriptor(path) for name, path in paths.items()},
        "accepted_bet_decision_freeze": _descriptor(decision_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "future_target_access_allowed": support["target_free_support_gate_passed"],
        "side_count_hard_gate_enabled": False,
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_score_mutated": False,
        **_blocked_confirmatory_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_confirmatory_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, report, manifest_path, manifest)


def run_v6_9_confirmatory_post_freeze(
    config: V69ConfirmatoryPostFreezeConfig,
    *,
    provider_factory: Callable[[], Any] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_ms_fn: Callable[[], int] = lambda: int(time.time() * 1000),
) -> dict[str, Any]:
    """Settle or evaluate the frozen v6.9 confirmatory window."""

    inputs, profile, freeze = _verified_post_freeze_inputs(config)
    if config.stage == "settle":
        return _settle(
            config,
            inputs=inputs,
            freeze=freeze,
            provider_factory=provider_factory,
            monotonic_fn=monotonic_fn,
            sleep_fn=sleep_fn,
            clock_ms_fn=clock_ms_fn,
        )
    runtime_profile, settled_index = _verified_target_inputs(
        config, inputs=inputs, profile=profile, freeze=freeze
    )
    return _evaluate(
        config,
        inputs=inputs,
        profile=profile,
        freeze=freeze,
        runtime_profile=runtime_profile,
        settled_index=settled_index,
    )


def _materialize_pinned_batch_artifacts(
    *,
    service_root: Path,
    selected_index_rows: list[dict[str, Any]],
    candidate_manifest_path: Path,
    collection_plan_path: Path,
) -> dict[str, Any]:
    selected_ids = {str(row["market_id"]) for row in selected_index_rows}
    selected_batches = {str(row["batch_id"]) for row in selected_index_rows}
    manifest_paths = sorted(
        service_root.glob(
            "v6_9_batch_canary_runs/*/v6_9_future_batch_action_liveness_manifest.json"
        )
    )
    by_batch: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in manifest_paths:
        manifest = _load_json(path)
        batch_id = str(manifest.get("batch_id") or "")
        if batch_id in selected_batches:
            if batch_id in by_batch:
                raise ValueError("#231 duplicate v6.9 batch canary manifest")
            by_batch[batch_id] = (path, manifest)
    if set(by_batch) != selected_batches:
        raise ValueError("#231 selected window is missing pinned batch canary artifacts")

    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_paths: list[Path] = []
    for batch_id in sorted(
        selected_batches,
        key=lambda value: min(
            int(row["sequence"]) for row in selected_index_rows if row["batch_id"] == value
        ),
    ):
        path, manifest = by_batch[batch_id]
        _verify_batch_manifest(
            manifest,
            candidate_manifest_path=candidate_manifest_path,
            collection_plan_path=collection_plan_path,
        )
        report = _load_json(Path(_verified_descriptor(manifest["report"], "v6.9 report")["path"]))
        if (
            report.get("batch_action_liveness_passed") is not True
            or report.get("labels_outcomes_or_pnl_opened") is not False
        ):
            raise ValueError("#231 selected batch liveness is not eligible")
        v6_2_manifest = _load_json(
            Path(_verified_descriptor(manifest["v6_2_batch_canary_manifest"], "v6.2 batch")["path"])
        )
        development = _load_json(
            Path(
                _verified_descriptor(
                    v6_2_manifest["development_batch_canary_manifest"], "development batch"
                )["path"]
            )
        )
        descriptors = {
            "feature_rows": development["feature_rows"],
            "action_rows": development["five_action_grid"],
            "prediction_rows": v6_2_manifest["mean_ev_scored_rows"],
            "candidate_rows": manifest["candidate_rows"],
            "base_selected_rows": manifest["v6_7_base_selected_rows"],
            "mapped_rows": manifest["mapped_rows"],
            "accepted_rows": manifest["guard_accepted_rows"],
            "legacy_replay_rows": v6_2_manifest["full_guard_replay"],
        }
        for name, descriptor in descriptors.items():
            rows = _load_jsonl(
                Path(_verified_descriptor(descriptor, f"#231 {name}")["path"])
            )
            collected[name].extend(
                row for row in rows if str(row.get("market_id") or "") in selected_ids
            )
        used_paths.append(path)

    for name, rows in collected.items():
        if name in {"base_selected_rows", "mapped_rows", "accepted_rows"}:
            rows.sort(key=lambda row: (int(row["decision_ts"]), str(row["market_id"])))
        else:
            rows.sort(
                key=lambda row: (
                    str(row.get("market_id") or ""),
                    int(row.get("decision_ts") or 0),
                    str(row.get("action") or row.get("executed_action") or ""),
                )
            )
    return {**collected, "batch_manifest_paths": used_paths}


def _verify_batch_manifest(
    manifest: dict[str, Any], *, candidate_manifest_path: Path, collection_plan_path: Path
) -> None:
    if (
        manifest.get("candidate_manifest") != _descriptor(candidate_manifest_path)
        or manifest.get("collection_plan") != _descriptor(collection_plan_path)
        or manifest.get("labels_outcomes_or_pnl_opened") is not False
        or manifest.get("source_score_mutated") is not False
    ):
        raise ValueError("#231 v6.9 batch manifest lineage or sealing mismatch")
    for field, expected in _blocked_confirmatory_safety_fields().items():
        if manifest.get(field) != expected:
            raise ValueError(f"#231 v6.9 batch safety mismatch: {field}")


def _validate_materialized_target_free_rows(
    rows: dict[str, Any], *, selected_ids: set[str]
) -> None:
    target_free = [
        *rows["feature_rows"],
        *rows["action_rows"],
        *rows["prediction_rows"],
        *rows["candidate_rows"],
        *rows["base_selected_rows"],
        *rows["mapped_rows"],
        *rows["accepted_rows"],
        *rows["legacy_replay_rows"],
    ]
    if _find_nonempty_fields(target_free, FORBIDDEN_TARGET_FIELDS):
        raise ValueError("#231 target-free freeze contains forbidden target fields")
    if any(
        int(row.get("max_input_ts") or 0) > int(row.get("decision_ts") or 0)
        for row in rows["feature_rows"] + rows["action_rows"] + rows["prediction_rows"]
    ):
        raise ValueError("#231 target-free freeze feature causality violation")
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows["action_rows"]:
        groups[(str(row["market_id"]), int(row["decision_ts"]))].add(str(row["action"]))
    feature_groups = {
        (str(row["market_id"]), int(row["decision_ts"])) for row in rows["feature_rows"]
    }
    if (
        not groups
        or set(groups) != feature_groups
        or any(actions != FIVE_ACTIONS for actions in groups.values())
        or {market_id for market_id, _ in groups} != selected_ids
    ):
        raise ValueError("#231 final five-action grid is incomplete")
    if {str(row["market_id"]) for row in rows["prediction_rows"]} != selected_ids:
        raise ValueError("#231 final prediction market coverage mismatch")
    for name in ("base_selected_rows", "mapped_rows", "accepted_rows"):
        market_ids = [str(row["market_id"]) for row in rows[name]]
        if len(market_ids) != len(set(market_ids)) or not set(market_ids).issubset(selected_ids):
            raise ValueError(f"#231 {name} market identity mismatch")


def _verified_freeze_inputs(config: V69ConfirmatoryFreezeConfig) -> dict[str, Path]:
    inputs = {
        "service_root": Path(config.service_root_path).resolve(),
        "candidate_manifest": Path(config.candidate_manifest_path).resolve(),
        "collection_plan": Path(config.collection_plan_path).resolve(),
        "collector_index": Path(config.collector_index_path).resolve(),
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
    }
    expected = {
        "candidate_manifest": config.expected_candidate_manifest_sha256,
        "collection_plan": config.expected_collection_plan_sha256,
        "collector_index": config.expected_collector_index_sha256,
        "evaluation_profile": config.expected_evaluation_profile_sha256,
    }
    for name, sha in expected.items():
        _verify_pin(inputs[name], sha, f"#231 {name}")
    if config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256:
        raise ValueError("#231 execution-PnL gate profile is not frozen")
    profile = _load_json(inputs["evaluation_profile"])
    validate_regime_emergent_pnl_v6_8_profile(profile)
    candidate = _load_json(inputs["candidate_manifest"])
    plan = _load_json(inputs["collection_plan"])
    _validate_candidate_manifest(candidate)
    validate_v6_9_future_collection_plan(
        plan,
        candidate_manifest=candidate,
        candidate_manifest_sha256=config.expected_candidate_manifest_sha256,
    )
    if _sha256_file(inputs["collector_index"]) != config.expected_collector_index_sha256:
        raise ValueError("#231 collector index changed during freeze input validation")
    return inputs


def _verified_post_freeze_inputs(
    config: V69ConfirmatoryPostFreezeConfig,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    inputs = {
        "evaluation_profile": Path(config.evaluation_profile_path).resolve(),
        "prediction_freeze": Path(config.prediction_freeze_manifest_path).resolve(),
    }
    if config.expected_evaluation_profile_sha256 != FROZEN_EVALUATION_PROFILE_SHA256:
        raise ValueError("#231 evaluation profile is not frozen")
    _verify_pin(
        inputs["evaluation_profile"],
        config.expected_evaluation_profile_sha256,
        "#231 evaluation profile",
    )
    _verify_pin(
        inputs["prediction_freeze"],
        config.expected_prediction_freeze_manifest_sha256,
        "#231 prediction freeze",
    )
    profile = _load_json(inputs["evaluation_profile"])
    validate_regime_emergent_pnl_v6_8_profile(profile)
    freeze = _load_json(inputs["prediction_freeze"])
    _validate_freeze(freeze, profile=profile, profile_path=inputs["evaluation_profile"])
    return inputs, profile, freeze


def _validate_freeze(
    freeze: dict[str, Any], *, profile: dict[str, Any], profile_path: Path
) -> None:
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("role") != "future_confirmatory"
        or freeze.get("evaluation_profile") != _descriptor(profile_path)
        or freeze.get("future_target_access_allowed") is not True
        or freeze.get("side_count_hard_gate_enabled") is not False
        or freeze.get("labels_outcomes_resolution_or_pnl_opened") is not False
        or freeze.get("settlement_provider_called") is not False
        or freeze.get("source_score_mutated") is not False
    ):
        raise ValueError("#231 target-free freeze is not eligible")
    for field, expected in _blocked_confirmatory_safety_fields().items():
        if freeze.get(field) != expected:
            raise ValueError(f"#231 target-free freeze safety mismatch: {field}")
    candidate_path = Path(
        _verified_descriptor(freeze["candidate_manifest"], "candidate manifest")["path"]
    )
    candidate = _load_json(candidate_path)
    plan_path = Path(_verified_descriptor(freeze["collection_plan"], "collection plan")["path"])
    plan = _load_json(plan_path)
    _validate_candidate_manifest(candidate)
    validate_v6_9_future_collection_plan(
        plan,
        candidate_manifest=candidate,
        candidate_manifest_sha256=freeze["candidate_manifest"]["sha256"],
    )
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_9_selected_decisions"], "v6.9 decisions")["path"])
    )
    decision = _load_json(
        Path(
            _verified_descriptor(
                freeze["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
            )["path"]
        )
    )
    support = build_regime_emergent_target_free_support(
        decisions,
        exact_window_market_count=len(selected),
        expected_window_market_count=WINDOW_MARKET_COUNT,
        required_total_market_count=int(
            profile["future_confirmatory"]["minimum_guard_accepted_unique_market_count_total"]
        ),
        score_field="v6_9_calibrated_runtime_expected_pnl_per_contract",
    )
    selected_ids = [str(row.get("market_id") or "") for row in selected]
    decision_ids = {str(row.get("market_id") or "") for row in decisions}
    boundary = int(candidate["candidate_freeze_created_ts"])
    prior_ids, _ = _prior_market_reference(candidate)
    if (
        len(selected) != WINDOW_MARKET_COUNT
        or "" in selected_ids
        or len(set(selected_ids)) != WINDOW_MARKET_COUNT
        or prior_ids.intersection(selected_ids)
        or any(int(row["scheduled_round_start_ts"]) <= boundary for row in selected)
        or "" in decision_ids
        or not decision_ids.issubset(set(selected_ids))
        or decision.get("selected_window_market_ids") != selected_ids
        or decision.get("target_free_support") != support
        or support["target_free_support_gate_passed"] is not True
        or decision.get("future_target_access_allowed") is not True
    ):
        raise ValueError("#231 target-free decision-freeze evidence mismatch")


def _verified_target_inputs(
    config: V69ConfirmatoryPostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_path = Path(config.runtime_policy_profile_path).resolve()
    index_path = Path(config.settled_corpus_index_path).resolve()
    if (
        config.expected_runtime_policy_profile_sha256 != FROZEN_RUNTIME_POLICY_PROFILE_SHA256
        or config.expected_runtime_policy_profile_sha256
        != profile["lineage"]["runtime_policy_profile_sha256"]
    ):
        raise ValueError("#231 runtime target contract is not frozen")
    _verify_pin(
        runtime_path,
        str(config.expected_runtime_policy_profile_sha256),
        "#231 runtime profile",
    )
    _verify_pin(
        index_path,
        str(config.expected_settled_corpus_index_sha256),
        "#231 settled index",
    )
    runtime_profile = _load_json(runtime_path)
    validate_runtime_aligned_sbc_net_return_v6_4_profile(runtime_profile)
    if runtime_policy_source_hashes() != runtime_profile["runtime_policy_contract"][
        "source_function_sha256"
    ]:
        raise ValueError("#231 runtime policy source hashes drifted")
    settled_index = _load_json(index_path)
    _validate_settled_index(
        settled_index,
        freeze=freeze,
        freeze_path=inputs["prediction_freeze"],
        evaluation_started_ts=config.stage_started_ts,
    )
    inputs["runtime_profile"] = runtime_path
    inputs["settled_index"] = index_path
    return runtime_profile, settled_index


def _settle(
    config: V69ConfirmatoryPostFreezeConfig,
    *,
    inputs: dict[str, Path],
    freeze: dict[str, Any],
    provider_factory: Callable[[], Any] | None,
    monotonic_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    clock_ms_fn: Callable[[], int],
) -> dict[str, Any]:
    from bigan.v8.polymarket.recorder import PolymarketPublicHTTPRealCorpusProvider

    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "accepted-bet decision freeze"
    )
    decision = _load_json(Path(decision_descriptor["path"]))
    if config.stage_started_ts <= int(decision["decision_freeze_created_ts"]):
        raise ValueError("#231 settlement attempted before decision freeze")
    if config.stage_started_ts <= max(int(row["market_end_ts"]) for row in selected):
        raise ValueError("#231 settlement attempted before markets closed")
    frozen_features: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(
        Path(_verified_descriptor(freeze["target_free_feature_rows"], "features")["path"])
    ):
        frozen_features[str(row["market_id"])].append(row)

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    (run_dir / "settled_round_copies").mkdir()
    (run_dir / "settled_corpus_quarantine").mkdir()
    marker = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-start-marker-v1",
        "run_id": config.run_id,
        "role": "future_confirmatory",
        "target_access_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "all_markets_closed_before_target_access": True,
        "official_read_only_resolution_only": True,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_applied": False,
        **_blocked_confirmatory_safety_fields(),
    }
    marker["marker_id"] = canonical_json_sha256(marker)
    marker_path = run_dir / "v6_9_confirmatory_settlement_started.json"
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
        raise ValueError("#231 settlement finalization precedes target access")
    index_path = run_dir / "v6_9_confirmatory_settled_corpus_index.json"
    if complete:
        payload = {
            "schema_version": f"{SCHEMA_PREFIX}-settled-corpus-index-v1",
            "run_id": config.run_id,
            "role": "future_confirmatory",
            "target_access_started_ts": config.stage_started_ts,
            "index_finalized_ts": finalized_ts,
            "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
            "decision_freeze_sha256": decision_descriptor["sha256"],
            "entry_count": len(entries),
            "entries": entries,
            "outcomes_used_for_decision_selection_or_tuning": False,
            "source_outcome_blind_rounds_mutated": False,
            "side_quota_applied": False,
            **_blocked_confirmatory_safety_fields(),
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
        "side_quota_applied": False,
        **_blocked_confirmatory_safety_fields(),
    }
    report["report_id"] = canonical_json_sha256(report)
    report_path = run_dir / "v6_9_confirmatory_settlement_report.json"
    _write_json(report_path, report)
    _write_text(report_path.with_suffix(".md"), _settlement_markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-settlement-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settlement_start_marker": _descriptor(marker_path),
        "settled_corpus_index": _descriptor(index_path) if complete else None,
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "settled_corpus_index_ready": complete,
        "source_outcome_blind_rounds_mutated": False,
        "side_quota_applied": False,
        **_blocked_confirmatory_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_confirmatory_settlement_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(
        run_dir,
        report_path,
        report,
        manifest_path,
        manifest,
        index_path if complete else None,
    )


def _evaluate(
    config: V69ConfirmatoryPostFreezeConfig,
    *,
    inputs: dict[str, Path],
    profile: dict[str, Any],
    freeze: dict[str, Any],
    runtime_profile: dict[str, Any],
    settled_index: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    claim_path = inputs["prediction_freeze"].parent / (
        "v6_9_confirmatory_single_use_target_claim.json"
    )
    claim = {
        "schema_version": f"{SCHEMA_PREFIX}-single-use-claim-v1",
        "run_id": config.run_id,
        "stage": "evaluate_confirmatory",
        "target_evaluation_started_ts": config.stage_started_ts,
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "result_selected_rerun_allowed": False,
        "side_quota_applied": False,
        **_blocked_confirmatory_safety_fields(),
    }
    claim["claim_id"] = canonical_json_sha256(claim)
    try:
        _write_single_use_claim(claim_path, claim)
    except Exception:
        shutil.rmtree(run_dir)
        raise
    decisions = _load_jsonl(
        Path(_verified_descriptor(freeze["v6_9_selected_decisions"], "v6.9 decisions")["path"])
    )
    candidate_targets = _runtime_targets_for_decisions(
        decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-candidate",
        role="future_confirmatory",
    )
    legacy_replay = _load_jsonl(
        Path(_verified_descriptor(freeze["matched_legacy_guard_replay"], "legacy replay")["path"])
    )
    predictions = _load_jsonl(
        Path(
            _verified_descriptor(
                freeze["v6_2_target_free_predictions"], "v6.2 predictions"
            )["path"]
        )
    )
    legacy_decisions = _legacy_guard_accepted_sbc_decisions(
        legacy_replay, predictions=predictions
    )
    legacy_targets = _runtime_targets_for_decisions(
        legacy_decisions,
        settled_entries=list(settled_index["entries"]),
        runtime_profile=runtime_profile,
        run_id=f"{config.run_id}-legacy",
        role="future_confirmatory",
    )
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    gate = build_v6_8_regime_emergent_confirmatory_gate(
        candidate_targets,
        matched_legacy_rows=legacy_targets,
        evaluation_market_ids=[str(row["market_id"]) for row in selected],
        profile=profile,
        decision_freeze_sha256=decision_descriptor["sha256"],
    )
    gate.update(
        {
            "schema_version": f"{SCHEMA_PREFIX}-execution-pnl-gate-report-v1",
            "candidate_name": CANDIDATE_NAME,
            "frozen_gate_definition_reused_unchanged": True,
            "side_and_action_metrics_diagnostic_only": True,
        }
    )
    gate["report_id"] = canonical_json_sha256(
        {key: value for key, value in gate.items() if key != "report_id"}
    )
    candidate_path = run_dir / "v6_9_confirmatory_candidate_runtime_targets.jsonl"
    legacy_path = run_dir / "v6_9_confirmatory_matched_legacy_runtime_targets.jsonl"
    report_path = run_dir / "v6_9_confirmatory_execution_pnl_gate_report.json"
    _write_jsonl(candidate_path, candidate_targets)
    _write_jsonl(legacy_path, legacy_targets)
    _write_json(report_path, gate)
    _write_text(report_path.with_suffix(".md"), _evaluation_markdown(gate))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-evaluation-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "evaluation_profile": _descriptor(inputs["evaluation_profile"]),
        "prediction_freeze_manifest": _descriptor(inputs["prediction_freeze"]),
        "settled_corpus_index": _descriptor(inputs["settled_index"]),
        "runtime_policy_profile": _descriptor(inputs["runtime_profile"]),
        "single_use_claim": _descriptor(claim_path),
        "candidate_runtime_targets": _descriptor(candidate_path),
        "matched_legacy_runtime_targets": _descriptor(legacy_path),
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_path.with_suffix(".md")),
        "confirmatory_execution_pnl_gate_passed": gate[
            "confirmatory_execution_pnl_gate_passed"
        ],
        "confirmatory_execution_pnl_gate_blocking_reason_codes": gate[
            "confirmatory_execution_pnl_gate_blocking_reason_codes"
        ],
        "side_and_action_metrics_diagnostic_only": True,
        **_blocked_confirmatory_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v6_9_confirmatory_execution_pnl_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report_path, gate, manifest_path, manifest)


def _validate_settled_index(
    index: dict[str, Any],
    *,
    freeze: dict[str, Any],
    freeze_path: Path,
    evaluation_started_ts: int,
) -> None:
    selected = _load_jsonl(
        Path(_verified_descriptor(freeze["selected_window_rows"], "selected window")["path"])
    )
    selected_ids = {str(row["market_id"]) for row in selected}
    entries = list(index.get("entries") or [])
    entry_ids = {str(row.get("market_id") or "") for row in entries}
    decision_descriptor = _verified_descriptor(
        freeze["accepted_bet_decision_freeze"], "decision freeze"
    )
    if (
        index.get("schema_version") != f"{SCHEMA_PREFIX}-settled-corpus-index-v1"
        or index.get("role") != "future_confirmatory"
        or index.get("prediction_freeze_manifest") != _descriptor(freeze_path)
        or index.get("decision_freeze_sha256") != decision_descriptor["sha256"]
        or int(index.get("entry_count") or 0) != WINDOW_MARKET_COUNT
        or len(entries) != WINDOW_MARKET_COUNT
        or entry_ids != selected_ids
        or "" in entry_ids
        or evaluation_started_ts <= int(index.get("index_finalized_ts") or 0)
        or index.get("outcomes_used_for_decision_selection_or_tuning") is not False
        or index.get("side_quota_applied") is not False
    ):
        raise ValueError("#231 settled corpus index is not eligible")
    for field, expected in _blocked_confirmatory_safety_fields().items():
        if index.get(field) != expected:
            raise ValueError(f"#231 settled-index safety mismatch: {field}")
    for entry in entries:
        if (
            entry.get("official_read_only_resolution") is not True
            or entry.get("source_outcome_blind_round_mutated") is not False
        ):
            raise ValueError("#231 settled entry violates quarantine contract")
        for name in ("feature_rows", "label_rows", "resolution_events"):
            _verified_descriptor(entry[name], f"settled {name}")


def _prepare_run_dir(output_dir: Path, run_id: str, *, overwrite: bool) -> Path:
    run_dir = output_dir.resolve() / run_id
    if run_dir.exists():
        if not overwrite:
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


def _side_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row.get("side") or row.get("selected_side") or "") for row in rows).items())
    )


def _freeze_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Confirmatory Target-Free Freeze",
            "",
            f"- window markets: `{report['selected_window_market_count']}`",
            f"- guard-accepted markets: `{report['guard_accepted_market_count']}`",
            f"- side composition (diagnostic): `{report['guard_accepted_side_count_diagnostic']}`",
            f"- target-free support passed: `{str(report['target_free_support_gate_passed']).lower()}`",
            f"- blockers: `{report['target_free_support_blocking_reason_codes']}`",
            "- side quota / side-count gate: `false / false`",
            "- labels/outcomes/PnL opened: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _settlement_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Confirmatory Read-Only Settlement",
            "",
            f"- settled markets: `{report['settled_corpus_ready_market_count']}`",
            f"- unresolved markets: `{report['unresolved_or_failed_market_count']}`",
            f"- index ready: `{str(report['settled_corpus_index_ready']).lower()}`",
            "- source outcome-blind rounds mutated: `false`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


def _evaluation_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v6.9 Confirmatory Execution-PnL Gate",
            "",
            f"- gate passed: `{str(report['confirmatory_execution_pnl_gate_passed']).lower()}`",
            f"- blockers: `{report['confirmatory_execution_pnl_gate_blocking_reason_codes']}`",
            f"- candidate total PnL: `{report['candidate_after_cost_pnl']}`",
            f"- legacy total PnL: `{report['matched_legacy_after_cost_pnl']}`",
            f"- candidate-minus-legacy PnL: `{report['candidate_minus_matched_legacy_after_cost_pnl']}`",
            "- side/action metrics: `diagnostic_only`",
            "- paper/live/write/wallet/capital unlock: `false`",
            "",
        ]
    )


__all__ = [
    "V69ConfirmatoryFreezeConfig",
    "V69ConfirmatoryPostFreezeConfig",
    "freeze_v6_9_confirmatory_window",
    "run_v6_9_confirmatory_post_freeze",
    "select_v6_9_confirmatory_index_rows",
]
