"""Hash-pinned target-free freeze pipeline for issue #241."""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigan.v8.polymarket.contracts import canonical_json_sha256
from bigan.v8.polymarket.training.execution_layer_v2_nested_boosted_action_value_v7_4_canary import (
    _canonicalize_target_free_sbc_rows,
)
from bigan.v8.polymarket.training.execution_layer_v2_outcome_blind_batch_canary import (
    _prepare_run_dir,
    _result,
)
from bigan.v8.polymarket.training.execution_layer_v2_p_up_semantic_compatibility_v6_7 import (
    _microstructure_blocking_reasons,
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
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7 import (
    MODEL_SCHEMA_VERSION,
    validate_rolling_origin_drift_adaptive_v7_7_profile,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_canary import (
    _score_window,
)
from bigan.v8.polymarket.training.execution_layer_v2_rolling_origin_drift_adaptive_action_value_v7_7_future_holdout import (
    EXACT_MARKET_COUNT,
    FORBIDDEN_TARGET_FIELDS,
    FROZEN_PLAN_SHA256,
    SCAN_CAP,
    _safety_fields,
    build_v7_7_target_free_holdout_freeze_report,
    materialize_guard_accepted_runtime_decisions,
    select_v7_7_future_holdout_window,
    validate_v7_7_future_holdout_plan,
)

SCHEMA_PREFIX = (
    "bigan-v8-rolling-origin-drift-adaptive-action-value-v7-7-future-holdout"
)


@dataclass(frozen=True, slots=True)
class V77FutureTargetFreeFreezeConfig:
    """Pinned inputs for the one authoritative target-free decision freeze."""

    run_id: str
    output_dir: Path | str
    plan_path: Path | str
    expected_plan_sha256: str
    collector_protocol_path: Path | str
    expected_collector_protocol_sha256: str
    collector_index_path: Path | str
    expected_collector_index_sha256: str
    excluded_attempt_rows_path: Path | str
    expected_excluded_attempt_rows_sha256: str
    historical_manifest_path: Path | str
    expected_historical_manifest_sha256: str
    prior_lineage_rows_path: Path | str
    expected_prior_lineage_rows_sha256: str
    prior_canary_index_path: Path | str
    expected_prior_canary_index_sha256: str
    development_batch_manifest_paths: tuple[Path | str, ...]
    expected_development_batch_manifest_sha256s: tuple[str, ...]
    v6_2_batch_manifest_paths: tuple[Path | str, ...]
    expected_v6_2_batch_manifest_sha256s: tuple[str, ...]
    implementation_commit: str
    stage_started_ts: int
    overwrite_existing: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id is required")
        if len(self.implementation_commit) != 40:
            raise ValueError("implementation_commit must be a Git SHA-1")
        if self.stage_started_ts <= 0:
            raise ValueError("stage_started_ts must be positive")
        path_fields = (
            "output_dir",
            "plan_path",
            "collector_protocol_path",
            "collector_index_path",
            "excluded_attempt_rows_path",
            "historical_manifest_path",
            "prior_lineage_rows_path",
            "prior_canary_index_path",
        )
        for name in path_fields:
            object.__setattr__(self, name, Path(getattr(self, name)))
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
        pin_fields = (
            "expected_plan_sha256",
            "expected_collector_protocol_sha256",
            "expected_collector_index_sha256",
            "expected_excluded_attempt_rows_sha256",
            "expected_historical_manifest_sha256",
            "expected_prior_lineage_rows_sha256",
            "expected_prior_canary_index_sha256",
        )
        for name in pin_fields:
            _require_sha256(str(getattr(self, name)), name=name)
        for name, values in (
            (
                "expected_development_batch_manifest_sha256s",
                self.expected_development_batch_manifest_sha256s,
            ),
            (
                "expected_v6_2_batch_manifest_sha256s",
                self.expected_v6_2_batch_manifest_sha256s,
            ),
        ):
            for value in values:
                _require_sha256(value, name=name)
        if (
            not self.development_batch_manifest_paths
            or len(self.development_batch_manifest_paths)
            != len(self.expected_development_batch_manifest_sha256s)
            or len(self.v6_2_batch_manifest_paths)
            != len(self.expected_v6_2_batch_manifest_sha256s)
            or len(self.development_batch_manifest_paths)
            != len(self.v6_2_batch_manifest_paths)
        ):
            raise ValueError("development/v6.2 batch manifest pins must be nonempty and aligned")


def run_v7_7_future_target_free_freeze(
    config: V77FutureTargetFreeFreezeConfig,
) -> dict[str, Any]:
    """Select exact-120, score both frozen policies, and seal decisions before targets."""

    paths = {
        "plan": config.plan_path.resolve(),
        "protocol": config.collector_protocol_path.resolve(),
        "index": config.collector_index_path.resolve(),
        "excluded_attempts": config.excluded_attempt_rows_path.resolve(),
        "historical": config.historical_manifest_path.resolve(),
        "prior_lineage": config.prior_lineage_rows_path.resolve(),
        "prior_canary_index": config.prior_canary_index_path.resolve(),
    }
    pins = {
        "plan": config.expected_plan_sha256,
        "protocol": config.expected_collector_protocol_sha256,
        "index": config.expected_collector_index_sha256,
        "excluded_attempts": config.expected_excluded_attempt_rows_sha256,
        "historical": config.expected_historical_manifest_sha256,
        "prior_lineage": config.expected_prior_lineage_rows_sha256,
        "prior_canary_index": config.expected_prior_canary_index_sha256,
    }
    for name, path in paths.items():
        _verify_pin(path, pins[name], f"#241 {name}")
    if pins["plan"].lower() != FROZEN_PLAN_SHA256:
        raise ValueError("#241 target-free freeze plan pin drifted")
    plan = _load_json(paths["plan"])
    validate_v7_7_future_holdout_plan(plan)
    if pins["protocol"].lower() != plan["lineage"]["collector_protocol_sha256"]:
        raise ValueError("#241 collector protocol pin drifted")
    canary_rows = load_and_validate_persistent_outcome_blind_index(
        paths["prior_canary_index"]
    )
    if (
        pins["prior_canary_index"].lower()
        != plan["lineage"]["target_free_canary_batch_index_sha256"]
        or not canary_rows
        or canary_rows[-1]["entry_sha256"]
        != plan["lineage"]["target_free_canary_batch_last_entry_sha256"]
    ):
        raise ValueError("#241 target-free canary boundary pin drifted")
    historical = _load_json(paths["historical"])
    _validate_historical_lineage(historical, plan=plan, manifest_sha256=pins["historical"])

    run_dir = _prepare_run_dir(
        Path(config.output_dir), config.run_id, overwrite=config.overwrite_existing
    )
    index_snapshot = run_dir / "v7_7_future_collector_index_snapshot.jsonl"
    shutil.copyfile(paths["index"], index_snapshot)
    if _sha256_file(index_snapshot) != pins["index"].lower():
        raise ValueError("#241 collector index changed while snapshotting")
    index_rows = load_and_validate_persistent_outcome_blind_index(index_snapshot)
    excluded_attempts = _load_and_validate_excluded_attempts(
        paths["excluded_attempts"], plan=plan
    )
    prior = _prior_reference_sets(
        prior_lineage_rows_path=paths["prior_lineage"],
        prior_canary_index_path=paths["prior_canary_index"],
        historical=historical,
    )
    selected, attempted, selection = select_v7_7_future_holdout_window(
        [*index_rows, *excluded_attempts],
        plan=plan,
        prior_market_ids=prior["market_ids"],
        prior_slugs=prior["slugs"],
        prior_decision_ids=prior["decision_ids"],
        prior_source_row_hashes=prior["source_row_hashes"],
    )
    if selection["exact_window_ready"] is not True:
        raise ValueError("#241 exact-120 target-free window is not ready")
    for row in selected:
        _verify_index_raw_descriptors(row)

    development, v6_2 = _load_pinned_batch_manifests(config, plan=plan)
    action_rows, feature_rows, scored_rows = _load_batch_target_free_rows(
        development, v6_2
    )
    selected_ids = [str(row["market_id"]) for row in selected]
    selected_set = set(selected_ids)
    action_rows = [row for row in action_rows if str(row.get("market_id")) in selected_set]
    feature_rows = [row for row in feature_rows if str(row.get("market_id")) in selected_set]
    scored_rows = [row for row in scored_rows if str(row.get("market_id")) in selected_set]
    forbidden = sorted(
        set(_find_nonempty_fields(action_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(feature_rows, FORBIDDEN_TARGET_FIELDS))
        | set(_find_nonempty_fields(scored_rows, FORBIDDEN_TARGET_FIELDS))
    )
    if forbidden:
        raise ValueError("#241 target-free inputs contain targets: " + ",".join(forbidden))

    model_descriptor = _verified_descriptor(historical["model"], "#241 model")
    profile_descriptor = _verified_descriptor(historical["profile"], "#241 v7.7 profile")
    v6_7_descriptor = _verified_descriptor(
        historical["v6_7_candidate_profile"], "#241 v6.7 profile"
    )
    v7_0_descriptor = _verified_descriptor(
        historical["v7_0_training_profile"], "#241 v7.0 profile"
    )
    model = _load_json(Path(model_descriptor["path"]))
    profile = _load_json(Path(profile_descriptor["path"]))
    v6_7_profile = _load_json(Path(v6_7_descriptor["path"]))
    v7_0_profile = _load_json(Path(v7_0_descriptor["path"]))
    validate_rolling_origin_drift_adaptive_v7_7_profile(profile)
    validate_p_up_semantic_compatibility_v6_7_profile(v6_7_profile)

    v6_7_candidates, candidate_summary = build_v6_7_target_free_candidate_rows(
        scored_rows,
        action_rows=action_rows,
        profile=v6_7_profile,
    )
    baseline_rows = select_v6_7_target_free_rows(
        v6_7_candidates, profile=v6_7_profile
    )
    canonical_rows, canonical_summary = _canonicalize_target_free_sbc_rows(
        scored_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
        v7_0_profile=v7_0_profile,
    )
    decisions, candidate_guard = _score_window(
        selected_ids,
        canonical_rows=canonical_rows,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        model=model,
        v6_7_profile=v6_7_profile,
    )
    baseline_guard = _baseline_guard_window(
        selected_ids,
        baseline_rows=baseline_rows,
        action_rows=action_rows,
        v6_7_profile=v6_7_profile,
    )
    report = build_v7_7_target_free_holdout_freeze_report(
        selected,
        attempted_rows=attempted,
        action_rows=action_rows,
        candidate_guard_rows=candidate_guard,
        baseline_guard_rows=baseline_guard,
        selection_summary=selection,
        plan=plan,
        stage_started_ts=config.stage_started_ts,
        collector_index_sha256=pins["index"],
    )
    candidate_runtime = materialize_guard_accepted_runtime_decisions(
        candidate_guard, action_rows=action_rows
    )
    baseline_runtime = materialize_guard_accepted_runtime_decisions(
        baseline_guard, action_rows=action_rows
    )
    return _write_freeze_artifacts(
        config,
        run_dir=run_dir,
        paths=paths,
        pins=pins,
        index_snapshot=index_snapshot,
        selected=selected,
        attempted=attempted,
        feature_rows=feature_rows,
        action_rows=action_rows,
        scored_rows=scored_rows,
        canonical_rows=canonical_rows,
        v6_7_candidates=v6_7_candidates,
        baseline_rows=baseline_rows,
        decisions=decisions,
        candidate_guard=candidate_guard,
        baseline_guard=baseline_guard,
        candidate_runtime=candidate_runtime,
        baseline_runtime=baseline_runtime,
        report=report,
        development=development,
        v6_2=v6_2,
        prior=prior,
        candidate_summary=candidate_summary,
        canonical_summary=canonical_summary,
        model_descriptor=model_descriptor,
        profile_descriptor=profile_descriptor,
        v6_7_descriptor=v6_7_descriptor,
        v7_0_descriptor=v7_0_descriptor,
    )


def _load_pinned_batch_manifests(
    config: V77FutureTargetFreeFreezeConfig, *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    development: list[dict[str, Any]] = []
    v6_2: list[dict[str, Any]] = []
    expected_source = str(plan["lineage"]["v6_2_source_candidate_manifest_sha256"])
    expected_feature_contract = str(plan["lineage"]["feature_contract_sha256"])
    for dev_path, dev_pin, score_path, score_pin in zip(
        config.development_batch_manifest_paths,
        config.expected_development_batch_manifest_sha256s,
        config.v6_2_batch_manifest_paths,
        config.expected_v6_2_batch_manifest_sha256s,
        strict=True,
    ):
        dev_path = dev_path.resolve()
        score_path = score_path.resolve()
        _verify_pin(dev_path, dev_pin, "#241 development batch manifest")
        _verify_pin(score_path, score_pin, "#241 v6.2 batch manifest")
        dev = _load_json(dev_path)
        score = _load_json(score_path)
        if (
            dev.get("development_data_canary_passed") is not True
            or dev.get("candidate_model_scoring_attempted") is not False
            or dev.get("labels_outcomes_or_pnl_opened") is not False
            or _verified_descriptor(dev["feature_contract"], "#241 feature contract")[
                "sha256"
            ]
            != expected_feature_contract
        ):
            raise ValueError("#241 development batch is not target-free eligible")
        if (
            score.get("labels_outcomes_or_pnl_opened") is not False
            or score.get("batch_id") != dev.get("batch_id")
            or _verified_descriptor(
                score["development_batch_canary_manifest"], "#241 matched development"
            )["sha256"]
            != dev_pin.lower()
            or _verified_descriptor(score["candidate_manifest"], "#241 v6.2 source")[
                "sha256"
            ]
            != expected_source
        ):
            raise ValueError("#241 v6.2 batch scoring lineage is invalid")
        development.append(dev)
        v6_2.append(score)
    batch_ids = [str(row.get("batch_id") or "") for row in development]
    if "" in batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("#241 batch identity is missing or duplicated")
    return development, v6_2


def _load_batch_target_free_rows(
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for dev, score in zip(development, v6_2, strict=True):
        action_rows.extend(
            _load_jsonl(
                Path(_verified_descriptor(dev["five_action_grid"], "#241 action grid")["path"])
            )
        )
        feature_rows.extend(
            _load_jsonl(
                Path(_verified_descriptor(dev["feature_rows"], "#241 feature rows")["path"])
            )
        )
        scored_rows.extend(
            _load_jsonl(
                Path(_verified_descriptor(score["mean_ev_scored_rows"], "#241 scored rows")["path"])
            )
        )
    return action_rows, feature_rows, scored_rows


def _load_and_validate_excluded_attempts(
    path: Path, *, plan: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load immutable unindexed attempts that still consume the frozen scan cap."""

    rows = _load_jsonl(path)
    boundary = int(
        plan["collection"]["strictly_later_minimum_market_start_ts_exclusive"]
    )
    attempt_ids: set[str] = set()
    for row in rows:
        attempt_id = str(row.get("attempt_id") or "")
        scheduled_ts = int(row.get("scheduled_round_start_ts") or 0)
        report = _verified_descriptor(
            row.get("pending_round_capture_report"), "#241 excluded capture report"
        )
        manifest = _verified_descriptor(
            row.get("pending_round_capture_manifest"),
            "#241 excluded capture manifest",
        )
        report_payload = _load_json(Path(report["path"]))
        manifest_payload = _load_json(Path(manifest["path"]))
        expected_row_id = canonical_json_sha256(
            {key: value for key, value in row.items() if key != "excluded_attempt_row_id"}
        )
        if (
            row.get("schema_version")
            != f"{SCHEMA_PREFIX}-excluded-collection-attempt-v1"
            or not attempt_id
            or attempt_id in attempt_ids
            or scheduled_ts <= boundary
            or row.get("capture_quality_valid") is not False
            or row.get("excluded_from_selection") is not True
            or row.get("excluded_from_settlement") is not True
            or row.get("excluded_from_quality_valid_support") is not True
            or row.get("counts_against_frozen_scan_cap") is not True
            or not row.get("capture_quality_reason_codes")
            or row.get("labels_outcomes_or_pnl_opened") is not False
            or row.get("settlement_finalizer_started") is not False
            or row.get("resolution_provider_called") is not False
            or row.get("market_start_ts") not in (None, 0)
            or row.get("excluded_attempt_row_id") != expected_row_id
            or report["path"] != str(Path(report["path"]).resolve())
            or manifest["path"] != str(Path(manifest["path"]).resolve())
        ):
            raise ValueError("#241 excluded collection attempt registry is invalid")
        expected_capture_fields = {
            "run_id": attempt_id,
            "capture_status": "blocked_fail_closed",
            "pending_resolution": False,
            "resolution_provider_called": False,
            "paper_only": True,
            "capital_at_risk": False,
            "polymarket_write_enabled": False,
            "wallet_signing_enabled": False,
            "live_exchange_write_enabled": False,
            "broker_exchange_write_enabled": False,
        }
        for payload in (report_payload, manifest_payload):
            if any(
                payload.get(field) != expected
                for field, expected in expected_capture_fields.items()
            ):
                raise ValueError(
                    "#241 excluded source capture safety or status is invalid"
                )
        if (
            report_payload.get("training_eligible") is not False
            or int(report_payload.get("raw_resolution_count") or 0) != 0
            or sorted(report_payload.get("public_collection_reason_codes") or [])
            != sorted(row["capture_quality_reason_codes"])
        ):
            raise ValueError("#241 excluded source capture evidence is invalid")
        for field, expected in _safety_fields().items():
            if row.get(field) != expected:
                raise ValueError(
                    f"#241 excluded collection attempt safety mismatch: {field}"
                )
        forbidden = _find_nonempty_fields(
            [row, report_payload, manifest_payload], FORBIDDEN_TARGET_FIELDS
        )
        if forbidden:
            raise ValueError("#241 excluded collection attempt contains targets")
        attempt_ids.add(attempt_id)
    return rows


def _validate_historical_lineage(
    historical: dict[str, Any], *, plan: dict[str, Any], manifest_sha256: str
) -> None:
    lineage = dict(plan["lineage"])
    model = _verified_descriptor(historical["model"], "#241 model")
    profile = _verified_descriptor(historical["profile"], "#241 profile")
    v6_7 = _verified_descriptor(
        historical["v6_7_candidate_profile"], "#241 v6.7 profile"
    )
    runtime = _verified_descriptor(
        historical["runtime_policy_profile"], "#241 runtime profile"
    )
    model_payload = _load_json(Path(model["path"]))
    if (
        manifest_sha256.lower() != lineage["historical_manifest_sha256"]
        or model["sha256"] != lineage["historical_model_sha256"]
        or profile["sha256"] != lineage["v7_7_profile_sha256"]
        or v6_7["sha256"] != lineage["v6_7_profile_sha256"]
        or runtime["sha256"] != lineage["runtime_policy_profile_sha256"]
        or historical.get("historical_noninferiority_gate_passed") is not True
        or historical.get("target_free_canary_collection_allowed") is not True
        or historical.get("fit_leakage_audit_passed") is not True
        or model_payload.get("schema_version") != MODEL_SCHEMA_VERSION
    ):
        raise ValueError("#241 frozen historical lineage is invalid")


def _prior_reference_sets(
    *,
    prior_lineage_rows_path: Path,
    prior_canary_index_path: Path,
    historical: dict[str, Any],
) -> dict[str, Any]:
    rows = _load_jsonl(prior_lineage_rows_path)
    canary_rows = load_and_validate_persistent_outcome_blind_index(
        prior_canary_index_path
    )
    consumed = _load_jsonl(
        Path(
            _verified_descriptor(
                historical["consumed_stream_five_action_rows"],
                "#241 consumed target-free action rows",
            )["path"]
        )
    )
    all_rows = [*rows, *canary_rows, *consumed]
    forbidden = _find_nonempty_fields(all_rows, FORBIDDEN_TARGET_FIELDS)
    if forbidden:
        raise ValueError("#241 prior identity registry opened targets")
    output = {
        "market_ids": {str(row.get("market_id") or "") for row in all_rows} - {""},
        "slugs": {str(row.get("slug") or "") for row in all_rows} - {""},
        "decision_ids": {str(row.get("decision_id") or "") for row in all_rows} - {""},
        "source_row_hashes": {
            str(row.get("source_row_hash") or "") for row in all_rows
        }
        - {""},
        "source_row_counts": {
            "historical_lineage": len(rows),
            "target_free_canary_index": len(canary_rows),
            "consumed_stream_action_rows": len(consumed),
        },
    }
    output["prior_reference_hash"] = canonical_json_sha256(
        {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in output.items()
        }
    )
    return output


def _baseline_guard_window(
    selected_market_ids: list[str],
    *,
    baseline_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    v6_7_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_by_market = {str(row["market_id"]): row for row in baseline_rows}
    source_by_key = {
        (str(row["market_id"]), int(row["decision_ts"]), str(row["action"])): row
        for row in action_rows
    }
    output = []
    for market_id in selected_market_ids:
        baseline = baseline_by_market.get(market_id)
        if baseline is None:
            action = "NO_TRADE"
            side = "NONE"
            reasons = ["v6_7_no_positive_guard_compatible_action"]
            source = None
        else:
            action = str(baseline["action"])
            side = "UP" if action.startswith("BUY_UP_") else "DOWN"
            source = source_by_key.get(
                (market_id, int(baseline["decision_ts"]), action)
            )
            reasons = (
                ["selected_action_source_row_missing"]
                if source is None
                else _microstructure_blocking_reasons(
                    source, guard=v6_7_profile["hard_execution_safety"]
                )
            )
        row = {
            "market_id": market_id,
            "decision_ts": int(source.get("decision_ts") or 0) if source else 0,
            "selected_action": action,
            "selected_side": side,
            "execution_guard_order_allowed": source is not None and not reasons,
            "execution_blocking_reason_codes": reasons,
            "p_up_action_disagreement_diagnostic_only": True,
            "p_up_side_alignment_filter_enabled": False,
            "pre_entry_market_exposure": 0.0,
            "same_market_position_exists": False,
            "same_side_position_exists": False,
            "full_execution_guard_unchanged": True,
            "source_score_mutated": False,
            "labels_outcomes_or_pnl_opened": False,
        }
        row["guard_replay_row_id"] = canonical_json_sha256(row)
        output.append(row)
    return output


def _write_freeze_artifacts(
    config: V77FutureTargetFreeFreezeConfig,
    *,
    run_dir: Path,
    paths: dict[str, Path],
    pins: dict[str, str],
    index_snapshot: Path,
    selected: list[dict[str, Any]],
    attempted: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    v6_7_candidates: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    candidate_guard: list[dict[str, Any]],
    baseline_guard: list[dict[str, Any]],
    candidate_runtime: list[dict[str, Any]],
    baseline_runtime: list[dict[str, Any]],
    report: dict[str, Any],
    development: list[dict[str, Any]],
    v6_2: list[dict[str, Any]],
    prior: dict[str, Any],
    candidate_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
    model_descriptor: dict[str, Any],
    profile_descriptor: dict[str, Any],
    v6_7_descriptor: dict[str, Any],
    v7_0_descriptor: dict[str, Any],
) -> dict[str, Any]:
    outputs = {
        "attempted_rows": run_dir / "v7_7_future_attempted_rows.jsonl",
        "selected_rows": run_dir / "v7_7_future_selected_rows.jsonl",
        "feature_rows": run_dir / "v7_7_future_target_free_feature_rows.jsonl",
        "five_action_rows": run_dir / "v7_7_future_target_free_five_action_rows.jsonl",
        "v6_2_scored_rows": run_dir / "v7_7_future_v6_2_scored_rows.jsonl",
        "canonical_rows": run_dir / "v7_7_future_canonical_sbc_rows.jsonl",
        "v6_7_candidates": run_dir / "v7_7_future_v6_7_candidate_rows.jsonl",
        "v6_7_selected": run_dir / "v7_7_future_v6_7_selected_rows.jsonl",
        "v7_7_decisions": run_dir / "v7_7_future_decisions.jsonl",
        "v7_7_guard": run_dir / "v7_7_future_guard_replay.jsonl",
        "v6_7_guard": run_dir / "v7_7_future_v6_7_guard_replay.jsonl",
        "v7_7_runtime": run_dir / "v7_7_future_guard_accepted_runtime_decisions.jsonl",
        "v6_7_runtime": run_dir / "v7_7_future_v6_7_guard_accepted_runtime_decisions.jsonl",
    }
    payloads = {
        "attempted_rows": attempted,
        "selected_rows": selected,
        "feature_rows": feature_rows,
        "five_action_rows": action_rows,
        "v6_2_scored_rows": scored_rows,
        "canonical_rows": canonical_rows,
        "v6_7_candidates": v6_7_candidates,
        "v6_7_selected": baseline_rows,
        "v7_7_decisions": decisions,
        "v7_7_guard": candidate_guard,
        "v6_7_guard": baseline_guard,
        "v7_7_runtime": candidate_runtime,
        "v6_7_runtime": baseline_runtime,
    }
    for name, path in outputs.items():
        _write_jsonl(path, payloads[name])
    report = {
        **report,
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "stage_started_ts": config.stage_started_ts,
        "collector_index_snapshot_sha256": _sha256_file(index_snapshot),
        "selected_market_ids_sha256": canonical_json_sha256(
            [str(row["market_id"]) for row in selected]
        ),
        "prior_reference_hash": prior["prior_reference_hash"],
        "prior_reference_source_row_counts": prior["source_row_counts"],
        "unindexed_excluded_attempt_count": len(
            _load_jsonl(paths["excluded_attempts"])
        ),
        "unindexed_excluded_attempts_count_against_scan_cap": True,
        "v6_7_candidate_summary": candidate_summary,
        "canonical_mapping_summary": canonical_summary,
        "v7_7_guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in candidate_guard
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
        "v6_7_guard_blocking_reason_distribution": dict(
            sorted(
                Counter(
                    reason
                    for row in baseline_guard
                    for reason in row["execution_blocking_reason_codes"]
                ).items()
            )
        ),
    }
    report["report_id"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    report_path = run_dir / "v7_7_future_target_free_freeze_report.json"
    report_md_path = run_dir / "v7_7_future_target_free_freeze_report.md"
    _write_json(report_path, report)
    _write_text(report_md_path, _markdown(report))
    manifest = {
        "schema_version": f"{SCHEMA_PREFIX}-target-free-freeze-manifest-v1",
        "run_id": config.run_id,
        "implementation_commit": config.implementation_commit,
        "plan": _descriptor(paths["plan"]),
        "collector_protocol": _descriptor(paths["protocol"]),
        "source_collector_index": {
            "path": str(paths["index"]),
            "sha256": pins["index"].lower(),
        },
        "excluded_attempt_rows": _descriptor(paths["excluded_attempts"]),
        "collector_index_snapshot": _descriptor(index_snapshot),
        "historical_manifest": _descriptor(paths["historical"]),
        "prior_lineage_rows": _descriptor(paths["prior_lineage"]),
        "prior_canary_index": _descriptor(paths["prior_canary_index"]),
        "model": model_descriptor,
        "profile": profile_descriptor,
        "v6_7_profile": v6_7_descriptor,
        "v7_0_profile": v7_0_descriptor,
        "development_batch_manifests": [
            _descriptor(path.resolve()) for path in config.development_batch_manifest_paths
        ],
        "v6_2_batch_manifests": [
            _descriptor(path.resolve()) for path in config.v6_2_batch_manifest_paths
        ],
        **{name: _descriptor(path) for name, path in outputs.items()},
        "report": _descriptor(report_path),
        "report_markdown": _descriptor(report_md_path),
        "exact_market_count": EXACT_MARKET_COUNT,
        "maximum_scan_count": SCAN_CAP,
        "v7_7_guard_accepted_market_count": len(candidate_runtime),
        "v6_7_guard_accepted_market_count": len(baseline_runtime),
        "decision_freeze_created_ts": config.stage_started_ts,
        "decision_freeze_written_before_target_access": True,
        "target_free_freeze_passed": report["target_free_freeze_passed"],
        "future_target_access_allowed": report["future_target_access_allowed"],
        "labels_outcomes_resolution_or_pnl_opened": False,
        "settlement_provider_called": False,
        "source_scores_mutated": False,
        "threshold_or_model_tuning_performed": False,
        "future_results_used_for_tuning": False,
        "result_selected_rerun_allowed": False,
        **_safety_fields(),
    }
    manifest["manifest_id"] = canonical_json_sha256(manifest)
    manifest_path = run_dir / "v7_7_future_target_free_freeze_manifest.json"
    _write_json(manifest_path, manifest)
    return _result(run_dir, report, report_path, manifest, manifest_path)


def _markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# v7.7 Future Target-Free Freeze",
            "",
            f"- run: `{report['run_id']}`",
            f"- selected markets: `{report['selected_market_count']}`",
            f"- v7.7 guard-accepted markets: `{report['v7_7_guard_accepted_market_count']}`",
            f"- v6.7 guard-accepted markets: `{report['v6_7_guard_accepted_market_count']}`",
            f"- side quota enabled: `{str(report['side_quota_enabled']).lower()}`",
            f"- target-free freeze passed: `{str(report['target_free_freeze_passed']).lower()}`",
            f"- blockers: `{report['target_free_blocking_reason_codes']}`",
            "- labels/outcomes/resolution/PnL opened: `false`",
            "- settlement provider called: `false`",
            "- paper/live/write/wallet/capital/handoff remain blocked.",
            "",
        ]
    )
